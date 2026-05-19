"""
Rebuild a partial hf_ckpt/ from the intact DCP state in the same checkpoint dir.

Use when a training kill caught the HF save mid-flight (e.g. only one shard written).
The DCP `model/` dir is the source of truth; auxiliary files (config.json,
lingbotvla_cli.yaml, tokenizer) are copied from a known-good sibling checkpoint.

Usage:
    python scripts/salvage_dcp_to_hf.py \\
        --dcp_root output/r1pro_delta_dual/checkpoints/global_step_4760 \\
        --template_hf_ckpt output/r1pro_delta_dual/checkpoints/global_step_4080/hf_ckpt
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch  # noqa: F401  (must import torch before lerobot)

# Import these AFTER torch so the path resolution works
from lingbotvla.checkpoint import dcp_to_torch_state_dict
from lingbotvla.models import save_model_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp_root", required=True, help="checkpoint dir (containing model/, optimizer/, hf_ckpt/)")
    ap.add_argument("--template_hf_ckpt", required=True, help="a sibling hf_ckpt/ dir with valid aux files")
    ap.add_argument("--save_dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    dcp_root = Path(args.dcp_root)
    template = Path(args.template_hf_ckpt)
    out_hf = dcp_root / "hf_ckpt"

    assert (dcp_root / "model").is_dir(), f"missing DCP model/ at {dcp_root}"
    assert template.is_dir(), f"template missing: {template}"

    # 1) Clean any partial files in out_hf except subdirs (none expected)
    out_hf.mkdir(exist_ok=True)
    for p in out_hf.iterdir():
        if p.is_file():
            print(f"removing partial: {p.name}")
            p.unlink()

    # 2) Load DCP -> state_dict and save as sharded safetensors + index.json
    print(f"loading DCP from {dcp_root / 'model'} ...")
    state_dict = dcp_to_torch_state_dict(str(dcp_root))
    print(f"  loaded {len(state_dict)} tensors")
    print(f"writing safetensors -> {out_hf} (dtype={args.save_dtype})")
    save_model_weights(str(out_hf), state_dict, save_dtype=args.save_dtype)

    # 3) Copy aux files (everything except *.safetensors and index files) from template
    aux_files = [
        "config.json",
        "lingbotvla_cli.yaml",
        "added_tokens.json",
        "chat_template.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    print("copying aux files from template:")
    for name in aux_files:
        src = template / name
        if not src.exists():
            print(f"  WARN: template missing {name}")
            continue
        shutil.copy2(src, out_hf / name)
        print(f"  {name} ({src.stat().st_size} bytes)")

    # 4) Sanity report
    print()
    print(f"=== final contents of {out_hf} ===")
    for p in sorted(out_hf.iterdir()):
        sz = p.stat().st_size
        print(f"  {p.name}  {sz:,} bytes")


if __name__ == "__main__":
    main()
