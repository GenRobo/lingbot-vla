"""
Dataset sanity check + training readiness assessment for LingBot-VLA fine-tuning.
Works with both LeRobot v2.1 and v3.0 datasets (reads parquet files directly).

Checks
------
  1. Format & metadata          — version, fps, feature keys, meta file consistency
  2. Episode overview           — length distribution, idle episode detection
  3. Per-joint analysis         — value ranges, % frames moving, step-to-step delta
  4. Timestamp / FPS            — duplicate frames, large gaps, effective FPS per episode
  5. Action-state latency       — lag-curve analysis, optimal lag in frames and ms
  6. Training readiness         — LingBot-VLA specific assessment (action type, FPS, latency)

Outputs
-------
  <dataset>/sanity_report/ep_overview.png
  <dataset>/sanity_report/joint_analysis.png
  <dataset>/sanity_report/timestamp.png
  <dataset>/sanity_report/latency.png
  <dataset>/sanity_report/sample_episodes.png
  <dataset>/sanity_report/summary_report.txt

Usage
-----
  python scripts/dataset/data_sanity_check.py --dataset_path /path/to/dataset
"""

import argparse
import json
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────────
STILL_THRESHOLD   = 0.01   # rad — max joint delta below this → frame is "still"
IDLE_EP_THRESHOLD = 0.50   # fraction — episode flagged if idle% exceeds this
MAX_LAG           = 20     # frames to search for optimal action-state lag

# joints to include in latency analysis (must have matching state key)
LATENCY_PAIRS = [
    ("action.right_arm",  "observation.state.right_arm"),
    ("action.left_arm",   "observation.state.left_arm"),
    ("action.torso",      "observation.state.torso"),
]

COLOURS = plt.cm.tab10(np.linspace(0, 0.9, 10))

# ── console tags ───────────────────────────────────────────────────────────────
_PASS = "\033[92m[PASS]\033[0m"
_WARN = "\033[93m[WARN]\033[0m"
_FAIL = "\033[91m[FAIL]\033[0m"
_INFO = "\033[94m[INFO]\033[0m"


class Report:
    """Accumulates console lines + plain-text lines for the summary file."""
    def __init__(self):
        self._lines: list[str] = []

    def log(self, msg: str = ""):
        print(msg)
        # strip ANSI codes for the file
        clean = msg
        for code in ["\033[92m", "\033[93m", "\033[91m", "\033[94m", "\033[0m"]:
            clean = clean.replace(code, "")
        self._lines.append(clean)

    def check(self, condition: bool, msg: str, level: str = "fail") -> bool:
        tag = _PASS if condition else (_WARN if level == "warn" else _FAIL)
        self.log(f"  {tag} {msg}")
        return condition

    def info(self, msg: str):
        self.log(f"  {_INFO} {msg}")

    def section(self, title: str):
        self.log(f"\n{'═'*50}")
        self.log(f"  {title}")
        self.log(f"{'═'*50}")

    def save(self, path: Path):
        path.write_text("\n".join(self._lines))


# ── helpers ────────────────────────────────────────────────────────────────────

def load_episodes(dataset_path: Path) -> list[pd.DataFrame]:
    """
    Load all episodes as a list of per-episode DataFrames.
    Handles both LeRobot v2.1 (one parquet per episode) and v3.0 (multiple
    episodes consolidated into chunk files, split here by episode_index).
    """
    data_dir = dataset_path / "data"
    eps: list[pd.DataFrame] = []

    v21_files = []
    v30_files = []
    for chunk in sorted(data_dir.glob("chunk-*")):
        v21_files.extend(sorted(chunk.glob("episode_*.parquet")))
        v30_files.extend(sorted(chunk.glob("file-*.parquet")))

    if v21_files:
        for pq in v21_files:
            eps.append(pd.read_parquet(pq))
    elif v30_files:
        for pq in v30_files:
            df = pd.read_parquet(pq)
            for ep_idx in sorted(df["episode_index"].unique()):
                eps.append(df[df["episode_index"] == ep_idx].reset_index(drop=True))
    return eps


def to_array(series) -> np.ndarray:
    return np.array(series.tolist())


def idle_fraction(ep: pd.DataFrame, action_keys: list[str]) -> float:
    """Fraction of frames with no meaningful motion across all action joints."""
    deltas = []
    for k in action_keys:
        if k in ep.columns and "gripper" not in k:
            arr = to_array(ep[k])
            if len(arr) > 1:
                deltas.append(np.abs(np.diff(arr, axis=0)).max(axis=1))
    if not deltas:
        return 0.0
    max_delta = np.stack(deltas, axis=1).max(axis=1)
    return float((max_delta < STILL_THRESHOLD).mean())


# ── section 1: format ─────────────────────────────────────────────────────────

def check_format(R: Report, meta_dir: Path, info: dict):
    R.section("1. FORMAT & METADATA")

    version = info.get("codebase_version", "unknown")
    R.info(f"codebase_version : {version}")
    if version == "v3.0":
        R.check(True, "Dataset is v3.0 — compatible with LingBot-VLA.")
    else:
        R.check(False,
                f"Dataset is {version}; LingBot-VLA requires v3.0. "
                "Run: python scripts/dataset/convert_v21_to_v30.py --dataset_path <path>",
                level="fail")

    R.info(f"total_episodes : {info.get('total_episodes','?')}  "
           f"total_frames : {info.get('total_frames','?')}  "
           f"declared fps : {info.get('fps','?')}")

    features   = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    R.info(f"cameras : {video_keys}")
    R.check(len(video_keys) > 0, f"{len(video_keys)} camera stream(s) found.")

    if version == "v3.0":
        # v3.0 stores per-episode metadata as parquet files under meta/episodes/
        episodes_dir = meta_dir / "episodes"
        ep_pqs = list(episodes_dir.rglob("*.parquet")) if episodes_dir.exists() else []
        R.check(len(ep_pqs) > 0,
                f"meta/episodes/*.parquet present ({len(ep_pqs)} file(s)).")
        R.check((meta_dir / "tasks.parquet").exists(),
                "meta/tasks.parquet present.")
        R.check(not (meta_dir / "stats.json").exists(),
                "meta/stats.json absent (correctly removed in v3.0).", level="warn")
    else:
        R.check((meta_dir / "episodes_stats.jsonl").exists(),
                "meta/episodes_stats.jsonl present (v2.1 expected this).",
                level="warn")

    return info.get("fps", 10), features, video_keys


# ── section 2: episode overview ───────────────────────────────────────────────

def check_episodes(R: Report, episodes: list[pd.DataFrame],
                   action_keys: list[str], out_dir: Path):
    R.section("2. EPISODE OVERVIEW  (length + idle detection)")

    lengths     = np.array([len(ep) for ep in episodes])
    idle_fracs  = np.array([idle_fraction(ep, action_keys) for ep in episodes])
    idle_eps    = np.where(idle_fracs > IDLE_EP_THRESHOLD)[0]

    R.info(f"episodes : {len(lengths)}")
    R.info(f"length   : min={lengths.min()}  max={lengths.max()}  "
           f"mean={lengths.mean():.1f}  std={lengths.std():.1f}")
    R.check(lengths.min() >= 10,
            f"{(lengths < 10).sum()} episodes shorter than 10 frames.", level="warn")

    p99 = np.percentile(lengths, 99)
    n_long = (lengths > p99).sum()
    R.check(n_long == 0,
            f"{n_long} episodes above 99th-pct length ({p99:.0f} frames) "
            "— possible merged/stuck recordings.", level="warn")

    R.check(len(idle_eps) == 0,
            f"{len(idle_eps)} episodes with >{IDLE_EP_THRESHOLD*100:.0f}% idle frames "
            f"(ep idx: {idle_eps.tolist()}) — remove before training.", level="fail")

    for ei in idle_eps:
        ep_idx = int(episodes[ei]["episode_index"].iloc[0])
        R.info(f"  ep {ep_idx:04d}: {idle_fracs[ei]*100:.0f}% idle, "
               f"{len(episodes[ei])} frames")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(lengths, bins=40, color=COLOURS[0], edgecolor="white", lw=0.4)
    ax1.axvline(lengths.mean(), color="red", ls="--", label=f"mean={lengths.mean():.0f}")
    ax1.set(xlabel="Episode length (frames)", ylabel="Count",
            title="Episode length distribution")
    ax1.legend()

    bar_c = [("red" if idle_fracs[i] > IDLE_EP_THRESHOLD else COLOURS[0])
             for i in range(len(idle_fracs))]
    ax2.bar(range(len(idle_fracs)), idle_fracs * 100, color=bar_c, width=1.0)
    ax2.axhline(IDLE_EP_THRESHOLD * 100, color="red", ls="--",
                label=f"threshold {IDLE_EP_THRESHOLD*100:.0f}%")
    ax2.set(xlabel="Episode index", ylabel="Idle frames (%)",
            title=f"Per-episode idle fraction  (still threshold={STILL_THRESHOLD} rad)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "ep_overview.png", dpi=120)
    plt.close(fig)

    return lengths, idle_fracs


# ── section 3: per-joint analysis ─────────────────────────────────────────────

def check_joints(R: Report, episodes: list[pd.DataFrame],
                 features: dict, out_dir: Path) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    R.section("3. PER-JOINT ANALYSIS  (value ranges, movement, gripper)")

    action_keys  = [k for k in features if k.startswith("action.") and
                    features[k].get("dtype") != "video"]
    state_keys   = [k for k in features if k.startswith("observation.state.") and
                    features[k].get("dtype") != "video"]
    R.info(f"action keys : {action_keys}")

    # collect all frames
    all_data: dict[str, np.ndarray] = {}
    for k in action_keys + state_keys:
        rows = [to_array(ep[k]) for ep in episodes if k in ep.columns]
        all_data[k] = np.concatenate(rows, axis=0) if rows else np.empty((0,))

    # gripper coverage
    R.log("\n  ── grippers ──")
    for k in action_keys:
        if "gripper" not in k:
            continue
        arr = all_data[k].flatten()
        n_unique = len(np.unique(arr.round(4)))
        if n_unique > 1:
            R.check(True,
                    f"{k}: range [{arr.min():.4f}, {arr.max():.4f}], unique values={n_unique}")
        else:
            R.check(False,
                    f"{k}: range [{arr.min():.4f}, {arr.max():.4f}], unique values={n_unique}"
                    "  ← stuck at constant, check recording pipeline.", level="fail")

    # per-joint movement table
    R.log("\n  ── joint movement (action keys, excluding grippers) ──")
    R.log(f"  {'key':<28} {'dim':<5} {'min':>8} {'max':>8} {'mean':>8} "
          f"{'step_Δ mean':>11} {'moving%':>8}")
    R.log("  " + "─" * 82)

    joint_keys = [k for k in action_keys if "gripper" not in k]
    per_joint_moving: dict[str, list] = {}  # key → per-dim moving%

    for k in joint_keys:
        arr = all_data[k]             # (N, D)
        if arr.ndim == 1:
            arr = arr[:, None]
        D = arr.shape[1]
        per_joint_moving[k] = []

        # per-episode step deltas
        ep_deltas = []
        for ep in episodes:
            if k in ep.columns:
                a = to_array(ep[k])
                if len(a) > 1:
                    ep_deltas.append(np.abs(np.diff(a, axis=0)))
        all_deltas = np.concatenate(ep_deltas, axis=0) if ep_deltas else np.zeros((1, D))

        for d in range(D):
            col      = arr[:, d]
            delta_d  = all_deltas[:, d]
            moving_p = float((delta_d > STILL_THRESHOLD).mean()) * 100
            per_joint_moving[k].append(moving_p)
            R.log(f"  {k:<28} {d:<5} {col.min():>8.3f} {col.max():>8.3f} "
                  f"{col.mean():>8.3f} {delta_d.mean():>11.5f} {moving_p:>7.1f}%")

        # action-state signal check
        sk = k.replace("action.", "observation.state.")
        if sk in all_data and len(all_data[sk]) == len(arr):
            mean_diff = float(np.abs(arr - all_data[sk]).mean())
            ok = mean_diff > 0.01
            R.check(ok,
                    f"{k}: mean|action−state|={mean_diff:.5f} "
                    + ("← genuine motion signal." if ok else
                       "← near-zero, actions echo state. Verify recording."),
                    level="warn" if not ok else "fail")

    # plot: boxplots of action values + % moving
    fig, axes = plt.subplots(len(joint_keys), 2,
                             figsize=(14, 3 * len(joint_keys)))
    if len(joint_keys) == 1:
        axes = axes[None, :]

    for row, k in enumerate(joint_keys):
        arr = all_data[k]
        if arr.ndim == 1:
            arr = arr[:, None]
        D = arr.shape[1]
        labels = [f"dim{i}" for i in range(D)]

        ax_box, ax_mov = axes[row]
        ax_box.boxplot(arr, tick_labels=labels, patch_artist=True,
                       boxprops=dict(facecolor=COLOURS[row % 10], alpha=0.6))
        ax_box.set_title(k, fontsize=9)
        ax_box.set_ylabel("rad")
        ax_box.grid(axis="y", ls="--", alpha=0.4)

        moving_pct = per_joint_moving[k]
        bar_c = [("green" if p > 10 else "salmon") for p in moving_pct]
        ax_mov.bar(labels, moving_pct, color=bar_c, edgecolor="white", lw=0.4)
        ax_mov.axhline(10, color="gray", ls="--", lw=0.8)
        ax_mov.set_ylim(0, 105)
        ax_mov.set_ylabel("% frames moving")
        ax_mov.set_title(f"{k} — movement per dim", fontsize=9)
        ax_mov.grid(axis="y", ls="--", alpha=0.4)

    fig.suptitle("Per-joint action distribution and movement", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_analysis.png", dpi=120)
    plt.close(fig)

    return action_keys, joint_keys, all_data


# ── section 4: timestamps / FPS ───────────────────────────────────────────────

def check_timestamps(R: Report, episodes: list[pd.DataFrame],
                     declared_fps, out_dir: Path):
    R.section("4. TIMESTAMP / FRAME-RATE CONSISTENCY")

    nominal_dt = 1.0 / declared_fps
    all_dts, eff_fps = [], []
    dup_count = large_gap_count = 0

    for ep in episodes:
        ts   = ep["timestamp"].values.astype(float)
        dts  = np.diff(ts)
        all_dts.append(dts)
        dup_count        += (dts == 0).sum()
        large_gap_count  += (dts > 2 * nominal_dt).sum()
        dur = ts[-1] - ts[0]
        if dur > 0:
            eff_fps.append((len(ts) - 1) / dur)

    flat    = np.concatenate(all_dts)
    eff_fps = np.array(eff_fps)
    total   = len(flat)

    R.info(f"dt (s)       : min={flat.min():.4f}  max={flat.max():.4f}  "
           f"mean={flat.mean():.4f}  std={flat.std():.4f}")
    R.info(f"effective FPS: min={eff_fps.min():.2f}  max={eff_fps.max():.2f}  "
           f"mean={eff_fps.mean():.2f}  std={eff_fps.std():.3f}")

    dup_pct = 100 * dup_count / total
    gap_pct = 100 * large_gap_count / total

    R.check(dup_count == 0,
            f"{dup_count} duplicate timestamps (dt=0), {dup_pct:.1f}% of transitions. "
            "Parquet action chunks use frame indices (not timestamps) so data loading is unaffected. "
            "However video decoding seeks by timestamp — duplicate entries return the same image frame twice.",
            level="warn")
    R.check(large_gap_count == 0,
            f"{large_gap_count} large gaps (dt>{2*nominal_dt:.3f}s), {gap_pct:.1f}% of transitions. "
            "Parquet chunks are index-based so gaps don't mis-align action data. "
            "Video decoding may skip frames across the gap.", level="warn")
    R.check(eff_fps.std() < 0.5,
            f"Effective FPS std={eff_fps.std():.3f} — "
            + ("consistent." if eff_fps.std() < 0.5 else "high variance across episodes."),
            level="warn")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    clip = min(4 * nominal_dt, flat.max())
    ax1.hist(flat[flat <= clip], bins=80, color=COLOURS[0], edgecolor="white", lw=0.3)
    ax1.axvline(nominal_dt, color="red", ls="--", label=f"nominal {nominal_dt:.3f}s")
    ax1.set(xlabel="Δt between frames (s)", ylabel="Count",
            title=f"Frame interval  (dt=0: {dup_count}, >2×nominal: {large_gap_count})")
    ax1.legend()

    ax2.plot(eff_fps, marker="o", ms=2, lw=0.8, color=COLOURS[0])
    ax2.axhline(declared_fps, color="red", ls="--", label=f"declared {declared_fps} fps")
    ax2.set(xlabel="Episode index", ylabel="Effective FPS",
            title="Per-episode effective FPS")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "timestamp.png", dpi=120)
    plt.close(fig)

    return dup_count, large_gap_count, eff_fps


# ── section 5: action-state latency ───────────────────────────────────────────

def check_latency(R: Report, episodes: list[pd.DataFrame], out_dir: Path):
    R.section("5. ACTION-STATE LATENCY")
    R.log("  method: argmin_k  mean|action[t] − state[t+k]|  for k=0..20")

    results: dict[str, dict] = {}

    for ak, sk in LATENCY_PAIRS:
        if not any(ak in ep.columns and sk in ep.columns for ep in episodes):
            continue

        curves, best_ks, lag_ms_list = [], [], []
        for ep in episodes:
            if ak not in ep.columns or sk not in ep.columns or len(ep) < MAX_LAG + 3:
                continue
            a  = to_array(ep[ak])
            s  = to_array(ep[sk])
            ts = ep["timestamp"].values.astype(float)

            curve = np.array([
                np.abs(a - s).mean() if k == 0 else np.abs(a[:-k] - s[k:]).mean()
                for k in range(MAX_LAG + 1)
            ])
            best_k = int(np.argmin(curve))
            lag_ms = ((ts[best_k:] - ts[:-best_k]).mean() * 1000) if best_k > 0 else 0.0

            curves.append(curve)
            best_ks.append(best_k)
            lag_ms_list.append(lag_ms)

        if not curves:
            continue

        curves  = np.array(curves)
        best_ks = np.array(best_ks)
        lag_ms  = np.array(lag_ms_list)

        results[ak] = {"curves": curves, "best_ks": best_ks, "lag_ms": lag_ms,
                       "mean_curve": curves.mean(0), "std_curve": curves.std(0)}

        label = ak.split("action.")[-1]
        R.info(f"{label:<12}  lag={best_ks.mean():.1f}±{best_ks.std():.1f} frames "
               f"[{best_ks.min()},{best_ks.max()}]  "
               f"latency={lag_ms.mean():.0f}±{lag_ms.std():.0f} ms")

    # plot lag curves + boxplot
    valid = {k: v for k, v in results.items() if v}
    n = len(valid)
    if n == 0:
        return results

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    ks = np.arange(MAX_LAG + 1)
    for ax, (ak, res), col in zip(axes, valid.items(), COLOURS):
        mc, sc = res["mean_curve"], res["std_curve"]
        best_k = int(np.argmin(mc))
        ax.plot(ks, mc, color=col, lw=2)
        ax.fill_between(ks, mc - sc, mc + sc, color=col, alpha=0.2)
        ax.axvline(best_k, color="red", ls="--", lw=1.2,
                   label=f"optimal k={best_k} "
                         f"({res['lag_ms'].mean():.0f}ms)")
        ax.set(xlabel="Lag k (frames)",
               ylabel="Mean |action[t]−state[t+k]| (rad)",
               title=ak.split("action.")[-1])
        ax.legend(fontsize=8)
        ax.grid(ls="--", alpha=0.4)

    fig.suptitle("Action-to-state lag curves  (lower = better alignment)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "latency.png", dpi=120)
    plt.close(fig)

    return results


# ── section 6: training readiness ─────────────────────────────────────────────

def training_assessment(R: Report, info: dict, dup_count: int, large_gap_count: int,
                         eff_fps: np.ndarray, idle_fracs: np.ndarray,
                         latency_results: dict, action_keys: list[str],
                         all_data: dict[str, np.ndarray]):
    R.section("6. TRAINING READINESS  (LingBot-VLA specific)")

    declared_fps = info.get("fps", 10)
    version      = info.get("codebase_version", "unknown")

    R.log("""
  ── LingBot-VLA action type ──
  Default: subtract_state=False  →  ABSOLUTE joint positions.
  The model predicts where each joint should be, not how much to move.
  Set subtract_state=True per joint in the robot config for DELTA actions.
  Grippers and end-effectors always use absolute (enforced in the codebase).
""")

    # build rows
    rows = [
        # (question, pass?, note)
        ("Format ready (v3.0)?",
         version == "v3.0",
         f"Current: {version}. Run convert_v21_to_v30.py." if version != "v3.0" else "Ready."),

        ("Timestamp duplicates / gaps?",
         dup_count == 0 and large_gap_count == 0,
         f"{dup_count} dt=0 frames, {large_gap_count} large gaps. "
         "Action chunks use frame indices (not timestamps) so parquet data is unaffected. "
         "Video decoding may return duplicate images at dt=0 frames."),

        ("Effective FPS stable?",
         eff_fps.std() < 0.5,
         f"std={eff_fps.std():.3f} (mean={eff_fps.mean():.2f}). "
         + ("OK." if eff_fps.std() < 0.5 else
            "High variance may desync action chunks from declared FPS.")),

        ("Idle episodes removed?",
         (idle_fracs > IDLE_EP_THRESHOLD).sum() == 0,
         f"{(idle_fracs > IDLE_EP_THRESHOLD).sum()} episodes >{IDLE_EP_THRESHOLD*100:.0f}% idle. "
         "Remove — they teach the model to do nothing."),

        ("right_arm action signal?",
         True,   # assessed qualitatively below
         ""),

        ("left_arm action signal?",
         True,
         ""),

        ("Gripper data valid?",
         True,
         ""),
    ]

    # override with latency / joint data
    ra = latency_results.get("action.right_arm", {})
    la = latency_results.get("action.left_arm",  {})

    if ra:
        mean_lag_ms = ra["lag_ms"].mean()
        best_k      = int(np.argmin(ra["mean_curve"]))
        ok = best_k <= 10    # acceptable if < 10 frames
        rows[4] = ("right_arm action signal?", ok,
                   f"Latency ≈ {mean_lag_ms:.0f}ms ({best_k} frames). "
                   + ("Absolute action, compatible with LingBot-VLA." if ok else
                      "Very high latency — consider smaller action chunk."))

    if la:
        mean_diff = float(np.abs(
            np.concatenate([to_array(ep["action.left_arm"]) for ep in [] if "action.left_arm" in ep.columns] or [np.zeros((1,7))])
        ).mean())
        # Check from lag curve mean error at k=0
        err0 = la["mean_curve"][0]
        ok_la = err0 > 0.01
        rows[5] = ("left_arm action signal?", ok_la,
                   f"Mean|action−state| at k=0 = {err0:.5f}. "
                   + ("Signal present." if ok_la else
                      "Near-zero — actions echo state. Verify recording pipeline."))

    gripper_keys = [k for k in action_keys if "gripper" in k]
    stuck = [k for k in gripper_keys if len(np.unique(all_data[k].flatten().round(4))) == 1]
    gripper_ok = len(stuck) == 0
    rows[6] = ("Gripper data valid?",
               gripper_ok,
               "All grippers have movement signal." if gripper_ok else
               f"{len(stuck)}/{len(gripper_keys)} gripper(s) stuck at constant value: {stuck}. "
               "Must fix before training pick-and-place tasks.")

    # print table
    R.log(f"  {'Question':<35} {'Status':<8} Notes")
    R.log("  " + "─" * 90)
    for question, passed, note in rows:
        status = "PASS" if passed else "FAIL"
        tag    = _PASS if passed else _FAIL
        R.log(f"  {tag} {question:<35} {note}")

    R.log("""
  ── Recommended fix order ──
    1. Fix gripper recording — re-record or repair the data export
    2. Fix left_arm action signal — verify what is being logged as action
    3. Remove / trim idle episodes (ep idx with >50% still frames)
    4. De-duplicate timestamps (drop dt=0 frames) and clip large gaps
    5. Update tasks.jsonl with a descriptive task prompt
    6. Convert to v3.0:  python scripts/dataset/convert_v21_to_v30.py
    7. Create configs/robot_configs/<name>.yaml  (feature mapping)
    8. Run scripts/compute_norm.py to compute norm stats
    9. Train with:  bash train.sh tasks/vla/train_lingbotvla.py <config.yaml> ...
""")


# ── sample episode plot ────────────────────────────────────────────────────────

def plot_sample_episodes(episodes: list[pd.DataFrame], out_dir: Path):
    """Action vs state for right_arm across 3 representative episodes."""
    if not any("action.right_arm" in ep.columns for ep in episodes):
        return

    eps_by_len = sorted(episodes, key=len)
    n = len(eps_by_len)
    chosen = [eps_by_len[0], eps_by_len[n // 2], eps_by_len[-1]]
    titles = ["shortest", "median", "longest"]

    fig, axes = plt.subplots(3, 7, figsize=(22, 9), sharey="col")

    for row, (ep, title) in enumerate(zip(chosen, titles)):
        a  = to_array(ep["action.right_arm"])      # (T, 7)
        s  = to_array(ep["observation.state.right_arm"])
        ts = ep["timestamp"].values.astype(float)
        ts = ts - ts[0]
        ep_idx = int(ep["episode_index"].iloc[0])

        for dim in range(7):
            ax = axes[row, dim]
            c  = COLOURS[dim]
            ax.plot(ts, s[:, dim], color=c, lw=1.2, alpha=0.6, label="state")
            ax.plot(ts, a[:, dim], color=c, lw=1.2, ls="--", label="action")
            ax.fill_between(ts, s[:, dim], a[:, dim], color=c, alpha=0.12)
            ax.grid(ls="--", alpha=0.3)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"dim{dim}", fontsize=9)
            if dim == 0:
                ax.set_ylabel(f"ep{ep_idx:04d}\n({title}, {len(ep)}fr)", fontsize=7)
            if row == 2:
                ax.set_xlabel("t (s)", fontsize=7)

    axes[0, 0].legend(fontsize=6)
    fig.suptitle("right_arm — action (dashed) vs state (solid), 3 representative episodes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_episodes.png", dpi=120)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dataset sanity check + training readiness for LingBot-VLA")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir",   default="",
                        help="Output dir (default: <dataset>/sanity_report)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} does not exist.")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else dataset_path / "sanity_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    R = Report()
    R.log(f"Dataset : {dataset_path}")
    R.log(f"Output  : {out_dir}")

    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    R.log("\nLoading episodes …")
    episodes = load_episodes(dataset_path)
    R.log(f"Loaded {len(episodes)} episodes.\n")
    if len(episodes) == 0:
        R.log("ERROR: No episodes found. Expected parquet files under data/chunk-*/.")
        sys.exit(1)

    # run sections
    declared_fps, features, _ = check_format(R, dataset_path / "meta", info)

    action_keys, joint_keys, all_data = check_joints(R, episodes, features, out_dir)

    lengths, idle_fracs = check_episodes(R, episodes, action_keys, out_dir)

    dup_count, large_gap_count, eff_fps = check_timestamps(
        R, episodes, declared_fps, out_dir)

    latency_results = check_latency(R, episodes, out_dir)

    training_assessment(R, info, dup_count, large_gap_count,
                        eff_fps, idle_fracs, latency_results, action_keys, all_data)

    R.log("Generating sample episode plot …")
    plot_sample_episodes(episodes, out_dir)

    report_path = out_dir / "summary_report.txt"
    R.save(report_path)
    R.log(f"\nAll plots and report saved to: {out_dir}")


if __name__ == "__main__":
    main()
