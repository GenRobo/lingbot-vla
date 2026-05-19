"""
Post-process per-trajectory .npz dumps written by the patched open_loop_eval.py
to compute MSE/MAE over arbitrary subsets of action dimensions.

For each checkpoint dir, loads every <traj>.npz (containing `gt`, `pred`,
`action_keys`, `action_widths`), slices the columns by the requested action
feature names, and writes a re-aggregated summary alongside the original.

Usage:
    python scripts/reaggregate_eval.py \\
        --eval_dir eval/r1pro_delta_dual \\
        --keep action.right_arm action.right_gripper \\
        --out_name summary_right_only.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def slice_indices(action_keys: list[str], action_widths: list[int], keep: list[str]) -> list[int]:
    """Compute column indices in the concatenated gt/pred array for the kept features."""
    indices = []
    offset = 0
    for name, width in zip(action_keys, action_widths):
        if name in keep:
            indices.extend(range(offset, offset + width))
        offset += width
    if not indices:
        raise ValueError(f"None of {keep} matched {action_keys}")
    return indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--keep", nargs="+", required=True,
                    help="action feature names to keep, e.g. action.right_arm action.right_gripper")
    ap.add_argument("--out_name", default="summary_subset.json")
    args = ap.parse_args()

    root = Path(args.eval_dir)
    out_summary = {}
    print(f"keeping action features: {args.keep}")

    for step_dir in sorted(root.glob("step_*")):
        step = int(step_dir.name.split("_")[1])
        npz_files = sorted(step_dir.glob("*.npz"))
        if not npz_files:
            print(f"  step_{step}: no .npz files found, skipping")
            continue

        per_traj = []
        idxs_logged = None
        for npz in npz_files:
            data = np.load(npz, allow_pickle=False)
            gt = data["gt"]
            pred = data["pred"]
            action_keys = [s.item() if hasattr(s, "item") else str(s) for s in data["action_keys"]]
            action_widths = list(map(int, data["action_widths"]))
            idxs = slice_indices(action_keys, action_widths, args.keep)
            if idxs_logged is None:
                print(f"  step_{step}: action layout = {list(zip(action_keys, action_widths))}; "
                      f"keep idxs = {idxs} ({len(idxs)} / {gt.shape[1]} cols)")
                idxs_logged = idxs

            gt_sub = gt[:, idxs]
            pred_sub = pred[:, idxs]
            mse = float(np.mean((gt_sub - pred_sub) ** 2))
            mae = float(np.mean(np.abs(gt_sub - pred_sub)))
            traj_id = int(npz.stem)
            per_traj.append({"traj_id": traj_id, "mse": mse, "mae": mae})

        per_traj.sort(key=lambda r: r["traj_id"])
        mse_avg = float(np.mean([r["mse"] for r in per_traj]))
        mae_avg = float(np.mean([r["mae"] for r in per_traj]))
        out_summary[str(step)] = {
            "mse_avg": mse_avg,
            "mae_avg": mae_avg,
            "n_trajs": len(per_traj),
            "kept_action_features": args.keep,
            "per_traj": per_traj,
        }
        print(f"  step_{step}: mse_avg={mse_avg:.4f}  mae_avg={mae_avg:.4f}  n={len(per_traj)}")

    out_path = root / args.out_name
    out_path.write_text(json.dumps(out_summary, indent=2))
    print(f"\nwrote: {out_path}")

    # Compact table
    print()
    print(f"{'step':>6}  {'mse_avg':>10}  {'mae_avg':>10}")
    for step in sorted(out_summary, key=int):
        v = out_summary[step]
        print(f"{step:>6}  {v['mse_avg']:>10.4f}  {v['mae_avg']:>10.4f}")


if __name__ == "__main__":
    main()
