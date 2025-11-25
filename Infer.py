#!/usr/bin/env python3
"""
python UniSandBox/Infer.py \
  --model_path UniSandBox/results/checkpoints-math/0000010 \
  --output_dir path/to/output_dir \
  --files path/to/math_1.jsonl \
  --max_mem_per_gpu 80GiB \
  --num_gpus 8 
  \
  --think            # Optional: Enable think mode
  --overwrite        # Optional: Regenerate existing images
  --skip_existing    # Optional: Skip existing images (enabled by default)
"""
# Standard libraries
import os
import json
import argparse
import multiprocessing as mp
import time
import re  # Added for sanitizing subfolder names
from pathlib import Path
from typing import List, Dict, Any

# Delay heavy imports until after CUDA_VISIBLE_DEVICES is set per process

def prepare_metadata(input_file: str, output_dir: str, skip_existing: bool = True) -> List[Dict[str, Any]]:
    """Convert a single jsonl file into a metadata list ready for t2i."""
    metadatas: List[Dict[str, Any]] = []
    file_type = "mapping" if "mapping" in input_file.lower() else "math"
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            if file_type == "math":
                prompt_id = data.get("id") or data.get("ID") or data.get("Question_id")
                prompt = data["Question"]
                image_path = os.path.join(output_dir, f"{prompt_id}.png")
                if skip_existing and os.path.exists(image_path):
                    continue
                metadatas.append({
                    "Question": prompt,
                    "prompt_id": prompt_id,
                    "Answer": data.get("Answer", ""),
                    "ID": prompt_id,
                    "original_index": idx,
                    "output_dir": output_dir,
                })
            else:  # mapping
                base_id = data["ID"]
                for qtype in ("A", "B"):
                    q_key = f"Question_{qtype}"
                    if q_key not in data:
                        continue
                    prompt_id = f"{base_id}_{qtype}"
                    image_path = os.path.join(output_dir, f"{prompt_id}.png")
                    if skip_existing and os.path.exists(image_path):
                        continue
                    metadatas.append({
                        "Question": data[q_key],
                        "prompt_id": prompt_id,
                        "Answer": data.get("Answer", ""),
                        "ID": base_id,
                        "question_type": qtype,
                        "original_index": idx,  # position within original file
                        "output_dir": output_dir,
                    })
    return metadatas


def run_on_gpu(gpu_id: int, prompts: List[Dict[str, Any]], common_args: argparse.Namespace, tmp_jsonl_path: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import t2i AFTER limiting visible devices so torch only sees one GPU
    from t2i import main as t2i_main, parser as t2i_parser  # type: ignore

    # Write chunk metadata to a temporary json so t2i can read it
    tmp_dir = Path(os.environ.get("BAGEL_TEMP_DIR", "/tmp/bagel_inference"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    meta_file = tmp_dir / f"meta_{gpu_id}_{pid}_{timestamp}.json"
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    # Assemble t2i args
    sub_argv = [
        "--model_path", common_args.model_path,
        "--max_mem_per_gpu", common_args.max_mem_per_gpu,
        "--output_dir", common_args.output_dir,
        "--metadata_file", str(meta_file),
        "--jsonl_output_path", str(tmp_jsonl_path),
    ]
    if common_args.think:
        sub_argv.append("--think")
    if common_args.overwrite:
        sub_argv.append("--overwrite")

    sub_args = t2i_parser.parse_args(sub_argv)
    print(f"[GPU {gpu_id}] Processing {len(prompts)} prompts ...")
    t2i_main(sub_args)
    meta_file.unlink(missing_ok=True)
    print(f"[GPU {gpu_id}] Done.")


def consolidate_jsonl(tmp_paths: List[Path], final_path: Path):
    records: List[Dict[str, Any]] = []
    for p in tmp_paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        p.unlink(missing_ok=True)
    records.sort(key=lambda x: x["original_index"])
    with final_path.open("w", encoding="utf-8") as f:
        for rec in records:
            rec.pop("original_index", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Consolidated results saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified BAGEL inference launcher")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--files", nargs="+", required=True, help="Absolute paths to jsonl files")
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare all metadata
    all_prompts: List[Dict[str, Any]] = []
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"Warning: {file_path} not found, skipping.")
            continue
        # Use the file stem as subfolder name, but remove a trailing "_<digits>" pattern
        stem = Path(file_path).stem
        stem = re.sub(r"_(\d+)$", r"\1", stem)
        sub_out = output_dir / stem
        sub_out.mkdir(parents=True, exist_ok=True)
        all_prompts.extend(prepare_metadata(file_path, str(sub_out), args.skip_existing))
    if not all_prompts:
        print("No prompts to process. Exiting.")
        return

    num_gpus = args.num_gpus

    # Split prompts
    chunks = [all_prompts[i::num_gpus] for i in range(num_gpus)]
    processes = []
    tmp_jsonl_paths: List[Path] = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        tmp_jsonl = output_dir / f"tmp_{gpu_id}.jsonl"
        tmp_jsonl_paths.append(tmp_jsonl)
        p = mp.Process(target=run_on_gpu, args=(gpu_id, chunk, args, tmp_jsonl))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Consolidate think results
    if args.think:
        consolidate_jsonl(tmp_jsonl_paths, output_dir / "output.jsonl")

    print("All tasks completed.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
