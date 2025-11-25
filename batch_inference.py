#!/usr/bin/env python3
import os
import json
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import time

def prepare_metadata_for_file(input_file, output_dir, file_type, skip_existing=True):
    """
    Prepare metadata for a single jsonl file and return the prepared metadata list
    Skip entries that already have corresponding images (controlled by skip_existing parameter)
    """
    metadatas = []
    skipped_count = 0
    original_index = 0  # Track index
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            
            if file_type == 'math':
                # Math file: directly use id as filename (supports id, ID fields)
                prompt_id = data.get('id') or data.get('ID') or data.get('Question_id')
                prompt = data['Question']
                
                # Check if image already exists (skip only when skip_existing=True)
                if skip_existing:
                    image_path = os.path.join(output_dir, f"{prompt_id}.png")
                    if os.path.exists(image_path):
                        skipped_count += 1
                        original_index += 1  # Increment index even when skipping
                        continue  # Skip existing images
                
                metadatas.append({
                    'Question': prompt,
                    'prompt_id': prompt_id,
                    'Answer': data.get('Answer', ''),
                    'id': prompt_id,
                    'original_index': original_index
                })
                original_index += 1
            
            elif file_type == 'mapping':
                # Mapping file: need to process Question_A and Question_B
                base_id = data['ID']
                
                # Process Question_A
                if 'Question_A' in data:
                    prompt_id_a = f"{base_id}_A"
                    # Check if image already exists (skip only when skip_existing=True)
                    if skip_existing:
                        image_path_a = os.path.join(output_dir, f"{prompt_id_a}.png")
                        if os.path.exists(image_path_a):
                            skipped_count += 1
                        else:
                            metadatas.append({
                                'Question': data['Question_A'],
                                'prompt_id': prompt_id_a,
                                'Answer': data.get('Answer', ''),
                                'ID': data['ID'],
                                'question_type': 'A',
                                'original_index': original_index
                            })
                    else:
                        metadatas.append({
                            'Question': data['Question_A'],
                            'prompt_id': prompt_id_a,
                            'Answer': data.get('Answer', ''),
                            'ID': data['ID'],
                            'question_type': 'A',
                            'original_index': original_index
                        })
                    original_index += 1
                
                # Process Question_B
                if 'Question_B' in data:
                    prompt_id_b = f"{base_id}_B"
                    # Check if image already exists (skip only when skip_existing=True)
                    if skip_existing:
                        image_path_b = os.path.join(output_dir, f"{prompt_id_b}.png")
                        if os.path.exists(image_path_b):
                            skipped_count += 1
                        else:
                            metadatas.append({
                                'Question': data['Question_B'],
                                'prompt_id': prompt_id_b,
                                'Answer': data.get('Answer', ''),
                                'ID': data['ID'],
                                'question_type': 'B',
                                'original_index': original_index
                            })
                    else:
                        metadatas.append({
                            'Question': data['Question_B'],
                            'prompt_id': prompt_id_b,
                            'Answer': data.get('Answer', ''),
                            'ID': data['ID'],
                            'question_type': 'B',
                            'original_index': original_index
                        })
                    original_index += 1
    
    # Output statistics
    total_processed = len(metadatas) + skipped_count
    if skip_existing and skipped_count > 0:
        tqdm.write(f"  Skipped existing images: {skipped_count}/{total_processed}")
    elif not skip_existing:
        tqdm.write(f"  Will regenerate all images (skip_existing=False)")
    
    return metadatas

def run_inference(input_file, output_dir, model_path, think_mode=False, max_mem_per_gpu="80GiB", num_gpus=1, skip_existing=True):
    """
    Run inference for a single file
    """
    # Determine file type: only treat as mapping if filename contains 'mapping', otherwise treat as math
    filename = os.path.basename(input_file)
    filename_lower = filename.lower()
    file_type = 'mapping' if ('mapping' in filename_lower) else 'math'
    
    # Prepare metadata
    metadatas = prepare_metadata_for_file(input_file, output_dir, file_type, skip_existing)
    
    if not metadatas:
        tqdm.write(f"Warning: No valid data found in file {input_file}")
        return
    

    # Use custom temporary directory (if specified), otherwise use default temporary directory
    tmp_dir = os.environ.get('BAGEL_TEMP_DIR', '/tmp/bagel_inference')
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create temporary metadata file (use process ID and timestamp to ensure uniqueness)
    pid = os.getpid()
    timestamp = int(time.time() * 1000)  # Millisecond timestamp
    temp_metadata_file = os.path.join(tmp_dir, f"temp_metadata_{pid}_{timestamp}.json")
    with open(temp_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    
    # Build inference command
    if num_gpus > 1:
        cmd = [
            "python", "t2i_multi_gpu.py",
            "--model_path", model_path,
            "--max_mem_per_gpu", max_mem_per_gpu,
            "--output_dir", output_dir,
            "--metadata_file", temp_metadata_file,
            "--num_gpus", str(num_gpus),
            "--overwrite"
        ]
    else:
        cmd = [
            "python", "t2i.py",
            "--model_path", model_path,
            "--max_mem_per_gpu", max_mem_per_gpu,
            "--output_dir", output_dir,
            "--metadata_file", temp_metadata_file,
            "--overwrite"
        ]
    
    if think_mode:
        cmd.append("--think")
    
    # Run inference (show detailed error messages)
    try:
        # Display executed command
        tqdm.write(f"  Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Display all output information
        if result.stdout:
            tqdm.write(f"  Output: {result.stdout.strip()}")
        if result.stderr:
            tqdm.write(f"  Error: {result.stderr.strip()}")
            
    except subprocess.CalledProcessError as e:
        # Display detailed error information
        error_msg = f"Inference failed (exit code: {e.returncode})"
        if e.stdout:
            error_msg += f"\nStandard output: {e.stdout.strip()}"
        if e.stderr:
            error_msg += f"\nError output: {e.stderr.strip()}"
        raise Exception(error_msg)
    finally:
        # Clean up temporary files
        if os.path.exists(temp_metadata_file):
            os.remove(temp_metadata_file)

def main():
    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument("--model_path", type=str, default="/mnt/data/checkpoints/BAGEL-7B-MoT",
                        help="Model path")
    parser.add_argument("--max_mem_per_gpu", type=str, default="80GiB",
                        help="Maximum memory per GPU")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--output_dir", type=str, default="UniSandBox/inference_results",
                        help="Output directory")
    parser.add_argument("--files", nargs='+', 
                        default=[
                            "benchmark/mapping1_updated.jsonl",
                            "benchmark/mapping2_updated.jsonl",
                            "benchmark/mapping3_updated.jsonl"
                        ],
                        help="List of absolute paths to jsonl files for inference")
    parser.add_argument("--modes", nargs='+', choices=['normal', 'think'],
                        default=['normal', 'think'],
                        help="Inference mode list (normal: normal mode, think: think mode)")
    parser.add_argument("--skip_existing", action='store_true', default=True,
                        help="Skip existing images (default: True)")
    parser.add_argument("--no_skip_existing", dest='skip_existing', action='store_false',
                        help="Do not skip existing images, regenerate all images")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Use file list set by command line arguments (absolute paths)
    file_paths = [Path(file_path) for file_path in args.files]
    
    # Set inference modes based on command line arguments
    inference_modes = []
    for mode in args.modes:
        if mode == 'normal':
            inference_modes.append(("normal", False))
        elif mode == 'think':
            inference_modes.append(("think", True))
    
    # Prepare all tasks
    tasks = []
    for mode_name, think_mode in inference_modes:
        for file_path in file_paths:
            if file_path.exists():
                # Remove file extension to use as directory name
                file_name_without_ext = file_path.stem
                # Extract dataset type information from file path (if path contains test/train, etc.)
                dataset_type = "test"

                
                file_output_dir = output_dir / dataset_type / mode_name / file_name_without_ext
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                tasks.append({
                    'input_file': str(file_path),
                    'output_dir': str(file_output_dir),
                    'dataset_type': dataset_type,
                    'mode_name': mode_name,
                    'think_mode': think_mode,
                    'file_name': file_path.name,
                    'file_name_without_ext': file_name_without_ext
                })
            else:
                print(f"Warning: File {file_path} does not exist, skipping")
    
    total_tasks = len(tasks)
    print(f"Starting batch inference, total {total_tasks} tasks")
    print(f"File list: {[str(p) for p in file_paths]}")
    print(f"Inference modes: {args.modes}")
    print("=" * 70)
    
    # Use tqdm to create progress bar
    success_count = 0
    failed_tasks = []
    
    with tqdm(total=total_tasks, desc="Batch inference progress", unit="task", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for i, task in enumerate(tasks):
            # Update progress bar description
            current_desc = f"[{task['dataset_type']}/{task['mode_name']}/{task['file_name_without_ext']}]"
            pbar.set_description(f"Processing {current_desc}")
            
            try:
                start_time = time.time()
                
                run_inference(
                    input_file=task['input_file'],
                    output_dir=task['output_dir'],
                    model_path=args.model_path,
                    think_mode=task['think_mode'],
                    max_mem_per_gpu=args.max_mem_per_gpu,
                    num_gpus=args.num_gpus,
                    skip_existing=args.skip_existing
                )
                
                end_time = time.time()
                duration = end_time - start_time
                success_count += 1
                
                # Display success message
                tqdm.write(f"✓ Completed [{i+1}/{total_tasks}]: {current_desc} (duration: {duration:.1f}s)")
                
            except Exception as e:
                failed_tasks.append({
                    'task': current_desc,
                    'error': str(e)
                })
                tqdm.write(f"✗ Failed [{i+1}/{total_tasks}]: {current_desc} - Error: {e}")
            
            # Update progress bar
            pbar.update(1)
            
            # Display current statistics
            pbar.set_postfix({
                'Success': success_count,
                'Failed': len(failed_tasks),
                'Remaining': total_tasks - i - 1
            })
    
    print("=" * 70)
    print("All inference tasks completed!")
    print(f"Success: {success_count}/{total_tasks}")
    print(f"Failed: {len(failed_tasks)}/{total_tasks}")
    
    if failed_tasks:
        print("\nFailed tasks:")
        for i, failed in enumerate(failed_tasks, 1):
            print(f"  {i}. {failed['task']} - {failed['error']}")
    
    if success_count == total_tasks:
        print("\n🎉 All tasks completed successfully!")
    else:
        print(f"\n⚠️  {len(failed_tasks)} tasks failed, please check error messages")

if __name__ == "__main__":
    main() 