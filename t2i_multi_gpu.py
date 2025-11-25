import os
import json
import argparse
import torch
import multiprocessing
from pathlib import Path
import time

def run_on_gpu(gpu_id, prompts, args, tmp_jsonl_path):
    """
    A target function to run the inference script on a specific GPU.
    """
    # Set the current process to use only one GPU card
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Dynamically import the main inference script and its dependencies
    # This is done here to ensure it happens within the new process
    from t2i import main as t2i_main
    from t2i import parser as t2i_parser # Import the parser from the inference script

    # Create a temporary metadata file for the current chunk of prompts
    # Use custom temp directory if specified, otherwise use default temp directory
    temp_dir = os.environ.get('BAGEL_TEMP_DIR', '/tmp/bagel_inference')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Use process ID and timestamp to ensure uniqueness across different inference runs
    pid = os.getpid()
    timestamp = int(time.time() * 1000)  # milliseconds for more uniqueness
    tmp_metadata_path = os.path.join(temp_dir, f"tmp_metadata_{gpu_id}_{pid}_{timestamp}.json")
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    # Construct arguments for the inference script
    sub_args_list = [
        '--model_path', args.model_path,
        '--max_mem_per_gpu', args.max_mem_per_gpu,
        '--output_dir', args.output_dir,
        '--metadata_file', tmp_metadata_path,
        # Pass the path for the temporary JSONL output file
        '--jsonl_output_path', str(tmp_jsonl_path),
    ]
    if args.think:
        sub_args_list.append('--think')
    if args.overwrite:
        sub_args_list.append('--overwrite')

    sub_args = t2i_parser.parse_args(sub_args_list)

    print(f"[GPU {gpu_id}] Starting inference with {len(prompts)} prompts.")
    # Call the main inference function
    t2i_main(sub_args)
    print(f"[GPU {gpu_id}] Finished inference.")

    # Clean up the temporary metadata file
    os.remove(tmp_metadata_path)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU launcher for BAGEL model inference.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--think", action="store_true", help="Enable 'think' mode to generate detailed prompts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image files.")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read all prompts and pre-assign unique IDs and original index
    print("Preparing and pre-assigning IDs to prompts...")
    with open(args.metadata_file, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    for i, meta in enumerate(metadatas):
        # Assign a globally unique and ordered prompt_id
        meta['prompt_id'] = meta.get('prompt_id', f"image_{i:05d}")
        # Store the original index for final sorting
        meta['original_index'] = i

    # 2. Distribute prompts evenly across GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        raise ConnectionError("No GPUs detected. This script requires at least one GPU.")
    
    chunks = [metadatas[i::num_gpus] for i in range(num_gpus)]
    print(f"Distributing {len(metadatas)} prompts across {num_gpus} GPUs.")

    # 3. Launch parallel processes
    processes = []
    tmp_jsonl_paths = []
    for gpu_id, prompts_chunk in enumerate(chunks):
        if not prompts_chunk:
            continue
        # Define a unique temporary file for each process to write its results
        tmp_jsonl_path = output_dir / f"tmp_results_{gpu_id}.jsonl"
        tmp_jsonl_paths.append(tmp_jsonl_path)
        
        p = multiprocessing.Process(target=run_on_gpu, args=(gpu_id, prompts_chunk, args, tmp_jsonl_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All inference processes have completed.")

    # 4. Consolidate results if in 'think' mode
    if args.think:
        print("Consolidating and sorting results...")
        all_records = []
        for path in tmp_jsonl_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_records.append(json.loads(line))
                os.remove(path) # Clean up temporary file

        # Sort the records based on their original position
        all_records.sort(key=lambda x: x['original_index'])

        # Write the final, sorted output.jsonl file
        final_jsonl_path = output_dir / "output.jsonl"
        with open(final_jsonl_path, "w", encoding="utf-8") as f:
            for record in all_records:
                del record['original_index'] # Remove the temporary sorting key
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Successfully created consolidated results file at: {final_jsonl_path}")

if __name__ == "__main__":
    # Set start method for multiprocessing to prevent CUDA initialization issues
    multiprocessing.set_start_method("spawn", force=True)
    main()