#!/bin/bash

MODEL_PATH="path/to/ckpt"
OUTPUT_DIR="path/to/output"
MAX_MEM="80GiB"
NUM_GPUS=8

FILES=(
  "path/to/math_1.jsonl"
  "path/to/math_2.jsonl"
  "path/to/math_3.jsonl"
  
)

python batch_inference.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_mem_per_gpu "$MAX_MEM" \
  --num_gpus "$NUM_GPUS" \
  --files "${FILES[@]}" \
  --modes normal

python batch_inference.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_mem_per_gpu "$MAX_MEM" \
  --num_gpus "$NUM_GPUS" \
  --files "${FILES[@]}" \
  --modes think 