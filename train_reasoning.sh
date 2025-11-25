#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
num_nodes=1       
node_rank=0    
master_addr="127.0.0.1"  
master_port=29500       
cd UniSandBox
model_path="/path/to/BAGEL-7B-MoT"
export PYTHONPATH=UniSandBox:$PYTHONPATH

checkpoint_dir="results/checkpoints-math1"
mkdir -p "$checkpoint_dir"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/Math/math1_5k_nofilt.yaml \
  --model_path $model_path \
  --visual_gen True \
  --visual_und False \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_workers 1 \
  --expected_num_tokens 32768 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 16384\
  --save_mode epoch\
  --save_every 6.0\
  --total_epochs 6.0\
  --checkpoint_dir "$checkpoint_dir"

  # --save_mode epoch\
  # --save_every_epochs 2.0\
  #  --total_epochs 18.0\
