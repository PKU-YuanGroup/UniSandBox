#!/usr/bin/env python3
 
"""
python UniSandBox/tool/CKPT_Transfer.py \
  --src_dir path/checkpoints/BAGEL-7B-MoT \
  --target_dirs path/results/checkpoints-xxx
"""

import os, shutil, glob, argparse, concurrent.futures
import torch
from safetensors.torch import load_file, save_file


def copy_files(src_dir, tgt_dir, exclude_file):
    print(f"Copying to: {tgt_dir}")
    # Removed timing
    os.makedirs(tgt_dir, exist_ok=True)
    files = [(p, os.path.join(tgt_dir, os.path.basename(p)), os.path.basename(p))
             for p in glob.glob(os.path.join(src_dir, '*')) if os.path.isfile(p) and os.path.basename(p) != exclude_file]

    def cp(args):
        src, dst, name = args
        try:
            print(f"Start copy: {name}")
            shutil.copy2(src, dst)
            print(f"Copied: {name}")
            return True
        except Exception as e:
            print(f"Copy failed: {name}, error: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(files))) as ex:
        res = list(ex.map(cp, files))
    print(f"Copied {sum(res)}/{len(files)} files")


def to_bf16(path):
    print(f"Convert to bf16: {path}")
    # Removed timing
    try:
        sd = load_file(path)
        out = {k: (t.to(torch.bfloat16) if t.dtype in [torch.float32, torch.float16] else t) for k, t in sd.items()}
        save_file(out, path)
        print(f"bf16 conversion done, converted {len(out)}/{len(sd)} tensors")
        return True
    except Exception as e:
        print(f"bf16 failed: {e}")
        return False


def merge(src_path, dst_path):
    print("\nStart merging ...")
    print(f"Source: {src_path}")
    print(f"Target: {dst_path}")
    # Removed size statistics
    try:
        src = load_file(src_path); dst = load_file(dst_path)
        miss = set(src) - set(dst)
        print(f"Missing params: {len(miss)}")
        if not miss:
            print("No merge needed"); return True
        # Removed timing
        for k in miss:
            dst[k] = src[k]
            print(f"Add param: {k}, shape={src[k].shape}")
        save_file(dst, dst_path)
        print(f"Merge done, added {len(miss)} params")
        return True
    except Exception as e:
        print(f"Merge error: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description='Copy, merge, bf16 convert safetensors.')
    ap.add_argument('--src_dir', default='', help='Source checkpoint directory')
    ap.add_argument('--target_dirs', default='', help='Target directories to be processed')
    ap.add_argument('--exclude_file', default='ema.safetensors')
    args = ap.parse_args()
    src_safetensors = os.path.join(args.src_dir, 'ema.safetensors')
    target_dirs_raw = args.target_dirs
    target_dirs = (target_dirs_raw.split(',') if isinstance(target_dirs_raw, str) else target_dirs_raw)
    target_dirs = [d for d in target_dirs if d]
    for tgt in target_dirs:
        print(f"\n==== Processing {tgt} ====")
        copy_files(args.src_dir, tgt, args.exclude_file)
        dst_path = os.path.join(tgt, 'model.safetensors')
        if os.path.exists(dst_path):
            merge(src_safetensors, dst_path)
            to_bf16(dst_path)
        else:
            print('model.safetensors not found, skip merge/convert')
    print("All finished")


if __name__ == '__main__':
    main() 

