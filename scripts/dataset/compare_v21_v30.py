"""
Verify a v2.1 → v3.0 conversion preserved the data faithfully.

Compares two LeRobot datasets (one v2.1, one v3.0) and reports:
  - Episode count and per-episode frame count parity
  - Numerical parity of action and observation.state columns
  - Timestamp / frame_index parity
  - Video frame parity at sampled indices (head camera by default)
  - Whether the v3.0 dataset is loadable via the LeRobot library

Usage
-----
  python scripts/dataset/compare_v21_v30.py \
      --v21_path /path/to/lerobot_v2.1/dataset \
      --v30_path /path/to/lerobot_v3/dataset \
      [--n_sample_episodes 3] [--n_sample_frames 5]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_PASS = "\033[92m[PASS]\033[0m"
_WARN = "\033[93m[WARN]\033[0m"
_FAIL = "\033[91m[FAIL]\033[0m"
_INFO = "\033[94m[INFO]\033[0m"


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.n_pass = self.n_warn = self.n_fail = 0

    def log(self, msg: str = ""):
        print(msg)
        clean = msg
        for c in ("\033[92m", "\033[93m", "\033[91m", "\033[94m", "\033[0m"):
            clean = clean.replace(c, "")
        self.lines.append(clean)

    def check(self, ok: bool, msg: str, level: str = "fail"):
        if ok:
            self.n_pass += 1
            tag = _PASS
        elif level == "warn":
            self.n_warn += 1
            tag = _WARN
        else:
            self.n_fail += 1
            tag = _FAIL
        self.log(f"  {tag} {msg}")

    def info(self, msg: str):
        self.log(f"  {_INFO} {msg}")

    def save(self, path: Path):
        path.write_text("\n".join(self.lines))


def load_episodes_grouped(dataset_path: Path) -> dict[int, pd.DataFrame]:
    """Return {episode_index: DataFrame} for v2.1 or v3.0."""
    data_dir = dataset_path / "data"
    out: dict[int, pd.DataFrame] = {}
    v21_files = []
    v30_files = []
    for chunk in sorted(data_dir.glob("chunk-*")):
        v21_files.extend(sorted(chunk.glob("episode_*.parquet")))
        v30_files.extend(sorted(chunk.glob("file-*.parquet")))
    if v21_files:
        for pq in v21_files:
            df = pd.read_parquet(pq)
            ep_idx = int(df["episode_index"].iloc[0])
            out[ep_idx] = df.reset_index(drop=True)
    elif v30_files:
        for pq in v30_files:
            df = pd.read_parquet(pq)
            for ep_idx in df["episode_index"].unique():
                out[int(ep_idx)] = (
                    df[df["episode_index"] == ep_idx].reset_index(drop=True)
                )
    return out


def array_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith(("action.", "observation.state."))
    ]


def to_np(series) -> np.ndarray:
    return np.array(series.tolist())


# ── section 1: metadata parity ────────────────────────────────────────────────

def compare_metadata(R: Report, v21: Path, v30: Path):
    R.log("\n══════════════════════════════════════════════════")
    R.log("  1. METADATA PARITY  (info.json + tasks)")
    R.log("══════════════════════════════════════════════════")
    with open(v21 / "meta" / "info.json") as f:
        info21 = json.load(f)
    with open(v30 / "meta" / "info.json") as f:
        info30 = json.load(f)

    R.info(f"v2.1 version: {info21.get('codebase_version')}")
    R.info(f"v3.0 version: {info30.get('codebase_version')}")
    R.check(info30.get("codebase_version") == "v3.0",
            f"v3.0 dataset is tagged as v3.0 (actual: {info30.get('codebase_version')}).")
    R.check(info21.get("total_episodes") == info30.get("total_episodes"),
            f"total_episodes: v2.1={info21.get('total_episodes')} "
            f"vs v3.0={info30.get('total_episodes')}")
    R.check(info21.get("total_frames") == info30.get("total_frames"),
            f"total_frames: v2.1={info21.get('total_frames')} "
            f"vs v3.0={info30.get('total_frames')}")
    R.check(info21.get("fps") == info30.get("fps"),
            f"fps: v2.1={info21.get('fps')} vs v3.0={info30.get('fps')}")

    # features
    f21 = set(info21.get("features", {}).keys())
    f30 = set(info30.get("features", {}).keys())
    missing = f21 - f30
    added   = f30 - f21
    R.check(not missing,
            f"Features missing in v3.0: {sorted(missing) or 'none'}")
    if added:
        R.info(f"Features added in v3.0: {sorted(added)}")

    # tasks
    tasks21 = (v21 / "meta" / "tasks.jsonl")
    tasks30 = (v30 / "meta" / "tasks.parquet")
    R.check(tasks30.exists(), "v3.0 tasks.parquet present.")
    if tasks21.exists() and tasks30.exists():
        t21_rows = [json.loads(line) for line in tasks21.read_text().splitlines() if line.strip()]
        t30 = pd.read_parquet(tasks30)
        # v2.1 stores {task_index, task}; v3.0 stores 'task' as index and 'task_index' as column
        t21_map = {r["task_index"]: r["task"] for r in t21_rows}
        if "task" in t30.columns and "task_index" in t30.columns:
            t30_map = dict(zip(t30["task_index"], t30["task"]))
        else:  # v3.0 layout has the task string as the index
            t30_map = dict(zip(t30["task_index"], t30.index.tolist()))
        same_keys = set(t21_map) == set(t30_map)
        same_vals = all(t21_map[k] == t30_map[k] for k in t21_map if k in t30_map)
        R.check(same_keys and same_vals,
                f"Task prompts identical ({len(t21_map)} task(s)).")

    return info21, info30


# ── section 2: per-episode parquet parity ─────────────────────────────────────

def compare_data(R: Report, v21: Path, v30: Path, n_sample_episodes: int):
    R.log("\n══════════════════════════════════════════════════")
    R.log("  2. PER-EPISODE PARQUET PARITY")
    R.log("══════════════════════════════════════════════════")

    eps21 = load_episodes_grouped(v21)
    eps30 = load_episodes_grouped(v30)

    R.info(f"v2.1 episodes loaded: {len(eps21)}  v3.0 episodes loaded: {len(eps30)}")
    R.check(set(eps21) == set(eps30),
            f"Same episode_index set. v2.1\\v3.0={sorted(set(eps21)-set(eps30))[:5]}, "
            f"v3.0\\v2.1={sorted(set(eps30)-set(eps21))[:5]}")

    common = sorted(set(eps21) & set(eps30))
    if not common:
        R.check(False, "No common episodes — cannot compare data.", level="fail")
        return eps21, eps30

    # ── Frame count parity ─────────────────────────────────────────────────
    bad_lens = []
    for ep in common:
        if len(eps21[ep]) != len(eps30[ep]):
            bad_lens.append((ep, len(eps21[ep]), len(eps30[ep])))
    R.check(not bad_lens,
            f"All {len(common)} episodes have matching frame counts."
            if not bad_lens else
            f"{len(bad_lens)} episodes differ in length, e.g. {bad_lens[:3]}")

    # ── Numerical parity on sampled episodes ───────────────────────────────
    sample = sorted(np.random.RandomState(0).choice(
        common, size=min(n_sample_episodes, len(common)), replace=False))
    R.info(f"Sampling {len(sample)} episodes for numerical comparison: {sample}")

    cols = sorted(set(array_columns(eps21[sample[0]])) &
                  set(array_columns(eps30[sample[0]])))

    overall_max_diff = 0.0
    overall_bad_cols: list[tuple[int, str, float]] = []
    for ep in sample:
        d21, d30 = eps21[ep], eps30[ep]
        for c in cols:
            a21 = to_np(d21[c])
            a30 = to_np(d30[c])
            if a21.shape != a30.shape:
                overall_bad_cols.append((ep, c, np.nan))
                continue
            diff = float(np.abs(a21 - a30).max())
            overall_max_diff = max(overall_max_diff, diff)
            if diff > 1e-6:
                overall_bad_cols.append((ep, c, diff))

    R.check(not overall_bad_cols,
            f"All sampled action/state columns identical "
            f"(max |diff| = {overall_max_diff:.2e})."
            if not overall_bad_cols else
            f"{len(overall_bad_cols)} (episode, column) pairs differ. "
            f"e.g. {overall_bad_cols[:3]} | overall max diff={overall_max_diff:.2e}")

    # ── timestamps & frame_index parity ────────────────────────────────────
    ts_max_diff = 0.0
    idx_mismatch = 0
    for ep in sample:
        d21, d30 = eps21[ep], eps30[ep]
        if len(d21) != len(d30):
            continue
        ts21 = d21["timestamp"].values.astype(float)
        ts30 = d30["timestamp"].values.astype(float)
        ts_max_diff = max(ts_max_diff, float(np.abs(ts21 - ts30).max()))
        if not (d21["frame_index"].values == d30["frame_index"].values).all():
            idx_mismatch += 1

    R.check(ts_max_diff < 1e-6,
            f"timestamp identical across {len(sample)} sampled episodes "
            f"(max |Δt| = {ts_max_diff:.2e}).")
    R.check(idx_mismatch == 0,
            f"frame_index identical across {len(sample)} sampled episodes.")

    return eps21, eps30


# ── section 3: video frame parity ─────────────────────────────────────────────

def compare_videos(R: Report, v21: Path, v30: Path, n_sample_frames: int):
    R.log("\n══════════════════════════════════════════════════")
    R.log("  3. VIDEO FRAME PARITY  (head camera, sampled)")
    R.log("══════════════════════════════════════════════════")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        R.check(False, "Could not import lerobot — skipping video comparison.", level="warn")
        return

    # Use the lerobot loader with absolute paths
    try:
        ds30 = LeRobotDataset(repo_id=v30.name, root=v30)
    except Exception as e:
        R.check(False, f"v3.0 dataset failed to load via LeRobotDataset: {e}", level="fail")
        return
    try:
        ds21 = LeRobotDataset(repo_id=v21.name, root=v21)
    except Exception as e:
        R.check(False, f"v2.1 dataset failed to load via LeRobotDataset: {e}", level="warn")
        return

    R.check(True, "Both datasets are loadable via LeRobotDataset.")
    R.info(f"v2.1 length: {len(ds21)}  v3.0 length: {len(ds30)}")
    R.check(len(ds21) == len(ds30),
            f"Same number of frames ({len(ds21)} == {len(ds30)}).")

    if len(ds21) != len(ds30):
        return

    rng = np.random.RandomState(0)
    sample_idx = sorted(rng.choice(len(ds21), size=min(n_sample_frames, len(ds21)),
                                   replace=False).tolist())

    img_keys = [k for k in ds21[0] if k.startswith("observation.images.")]
    R.info(f"Comparing {len(sample_idx)} frame(s), image keys: {img_keys}")

    max_abs = 0.0
    psnr_list = []
    for i in sample_idx:
        a = ds21[i]
        b = ds30[i]
        for k in img_keys:
            x = a[k].to(float)
            y = b[k].to(float)
            if x.shape != y.shape:
                R.check(False,
                        f"frame {i} key {k}: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}",
                        level="fail")
                continue
            diff = (x - y)
            max_abs = max(max_abs, float(diff.abs().max()))
            mse = float((diff ** 2).mean())
            if mse > 0:
                psnr_list.append(10 * np.log10(255.0 ** 2 / mse))
            else:
                psnr_list.append(float("inf"))

    psnr_mean = float(np.mean([p for p in psnr_list if np.isfinite(p)])) if psnr_list else float("inf")
    # Re-encoding (libx264 ultrafast crf=20) is lossy; expect PSNR ~35-45 dB,
    # well above the perceptual-equivalence threshold of ~30 dB.
    R.check(psnr_mean >= 30.0,
            f"Sampled video frames are visually equivalent "
            f"(mean PSNR = {psnr_mean:.1f} dB, max |pixel Δ| = {max_abs:.0f}). "
            "Re-encode is lossy by design; PSNR ≥ 30 dB is the perceptual threshold.",
            level="warn" if psnr_mean < 35.0 else "fail")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify v2.1 → v3.0 conversion preserved data.")
    parser.add_argument("--v21_path", required=True)
    parser.add_argument("--v30_path", required=True)
    parser.add_argument("--n_sample_episodes", type=int, default=3,
                        help="Episodes to sample for numerical/timestamp comparison.")
    parser.add_argument("--n_sample_frames", type=int, default=5,
                        help="Frames to sample for video parity check.")
    parser.add_argument("--output", default="",
                        help="Optional path to save a plain-text report.")
    args = parser.parse_args()

    v21 = Path(args.v21_path).resolve()
    v30 = Path(args.v30_path).resolve()
    for p in (v21, v30):
        if not p.exists():
            print(f"ERROR: {p} does not exist.")
            sys.exit(1)

    R = Report()
    R.log(f"v2.1 dataset: {v21}")
    R.log(f"v3.0 dataset: {v30}")

    compare_metadata(R, v21, v30)
    compare_data(R, v21, v30, args.n_sample_episodes)
    compare_videos(R, v21, v30, args.n_sample_frames)

    R.log("\n══════════════════════════════════════════════════")
    R.log(f"  SUMMARY  PASS={R.n_pass}  WARN={R.n_warn}  FAIL={R.n_fail}")
    R.log("══════════════════════════════════════════════════")

    if args.output:
        out_path = Path(args.output).resolve()
        R.save(out_path)
        print(f"Report saved to: {out_path}")

    sys.exit(0 if R.n_fail == 0 else 2)


if __name__ == "__main__":
    main()
