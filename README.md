<p align="center">
  <a href="https://arxiv.org/abs/2511.20561">
    <img
      src="https://img.shields.io/badge/UniSandbox-Paper-red"
      alt="UniSandbox Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/">
    <img 
        src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" 
        alt="UniSandbox Model"
    />
  </a>
  <a href="https://huggingface.co/datasets">
    <img 
        src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue" 
        alt="UniSandbox Data"
    />
  </a>
</p>

# UniSandbox: Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward

[](https://www.google.com/search?q=https://arxiv.org/abs/2501.xxxxx) [](https://opensaource.org/licenses/Apache-2.0)

> [Yuwei Niu¹²\*](https://purshow.github.io/), [Weiyang Jin³\*](https://github.com/WayneJin0918), Jiaqi Liao, Chaoran Feng¹, Peng Jin¹, Bin Lin¹, Zongjian Li¹, Bin Zhu¹, [Weihao Yu¹](https://whyu.me/), [Li Yuan¹⁴†](https://yuanli2333.github.io/)
>
> ¹Peking University, ²Chongqing University, ³HKU MMLab, ⁴PengCheng Laboratory  
> \*Equal contribution, †Corresponding Author
>
> **Contact:** niuyuwei04@gmail.com, yuanli-ece@pku.edu.cn
>
> We introduce **UniSandbox**, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: **reasoning generation** and **knowledge transfer**. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation.

-----

## 🛠️ Set up Environment

```bash
git clone https://github.com/PKU-YuanGroup/UniSandBox.git
cd UniSandBox
conda create -n unisandbox python=3.10 -y
conda activate unisandbox
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

> **Note:** The provided training scripts are configured for 8 GPUs (e.g., 8×A100 80G) via `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. You can adjust `NUM_GPUS`, memory, and device settings in the shell scripts to fit your hardware.
-----

## 🧠 Section 1: Reasoning Generation

This section focuses on evaluating and improving the model's ability to perform generation based on mathematical calculation or logical deduction using the **STARS** (Self-Training with Rejection Sampling) framework.

### 1\. Training (STARS)

To train the model using the STARS framework:

1. **Training data**
   - Download the reasoning splits (math and mapping) from [`Yuwei-Niu/UniSandbox`](https://huggingface.co/datasets/Yuwei-Niu/UniSandBox).
   - Make sure `data/dataset_info.py` correctly points to these folders (see the `math*_reject_5k` and `mapping*_1w_reject` entries).
2. **Benchmark JSONL**
   - Evaluation JSONL files for reasoning are stored in `benchmark/test_reasoning` (e.g., `math_1.jsonl`, `math_2.jsonl`, `mapping1.jsonl`, etc.). These are **only used for inference/evaluation**, not for training.
3. **Run training**

```bash
bash train_reasoning.sh   # default: Math (see --dataset_config_file in the script)
```

You can switch to other reasoning tasks (e.g., `math2`, `mapping1`) by changing `--dataset_config_file` in `train_reasoning.sh` to the corresponding YAML under `data/configs/Math` or `data/configs/Mapping`.
### 2\. Weight Completion & Conversion

After training, you must run the code to perform weight completion and precision conversion.

```bash
python tool/CKPT_Transfer.py \
  --src_dir path/to/checkpoints/BAGEL-7B-MoT \        # input: original checkpoint
  --target_dirs path/to/results/checkpoints-xxx      # your checkpoint_dir
```

### 3\. Inference

To generate images on the reasoning benchmarks:

1. **Edit `batch_run_eval.sh`**:
   - Set `MODEL_PATH` to your checkpoint_dir.
   - Set `OUTPUT_DIR` to where you want to save generated images.
   - Set `FILES` to the list of **absolute paths** of reasoning JSONL files, e.g.:

```bash
MODEL_PATH="/abs/path/to/results/checkpoints-math1"
OUTPUT_DIR="/abs/path/to/inference_results"
FILES=(
  "/abs/path/to/UniSandBox/benchmark/test_reasoning/math_1.jsonl"
  "/abs/path/to/UniSandBox/benchmark/test_reasoning/math_2.jsonl"
)
```

2. **Run batch inference** (normal + CoT/think mode):

```bash
bash batch_run_eval.sh
```

This script wraps `batch_inference.py` and will:

- Automatically detect the file type (math vs. mapping) from the filename.
- Generate images for both **normal** (no explicit CoT) and **think** (with CoT) modes.
- Organize results under `OUTPUT_DIR/test/{normal,think}/{jsonl_stem}/` (e.g., `.../test/normal/math_1/`).

For more advanced usage (e.g., single-GPU, custom memory limits, regenerating existing images), please refer to the inline comments and argument descriptions in `batch_inference.py`.

### 4\. Evaluation

We use **vLLM** and **Qwen2.5-VL-7B-Instruct** for evaluation.

**Prerequisites:**

  - Install vLLM (refer to the [vLLM Qwen2.5-VL docs](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html#gpu-deployment)).
  - Download the vision-language model [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

**Step 1: Launch the vLLM Server**

```bash
vllm serve path/to/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16
```

**Step 2: Run Evaluation Scripts**

Once the server is running, evaluate both math and mapping benchmarks. Example commands:

```bash
# Math reasoning (e.g., math_1)
python eval/eval_math.py \
  --jsonl benchmark/test_reasoning/math_1.jsonl \
  --image-dir /abs/path/to/inference_results/test/normal/math_1 \
  --model path/to/Qwen2.5-VL-7B-Instruct \
  --overwrite

# Symbolic mapping (e.g., mapping1)
python eval/eval_mapping.py \
  --jsonl benchmark/test_reasoning/mapping1.jsonl \
  --image-dir /abs/path/to/inference_results/test/normal/mapping1 \
  --model path/to/Qwen2.5-VL-7B-Instruct \
  --overwrite
```

The scripts will:

- Call the vLLM server with the **two-stage prompts**.
- Write `evaluation_results.csv` and `evaluation_results.log` into the corresponding `image-dir`.

For more detailed options (e.g., `--max-workers`, JSONL format), please check the top-of-file docstrings and inline comments in `eval/eval_math.py` and `eval/eval_mapping.py`.

-----

## 📚 Section 2: Knowledge Transfer

This section evaluates whether the model can utilize newly injected knowledge (e.g., virtual character profiles) for visual generation.

### 1\. Training (Knowledge Injection)

To inject new knowledge into the understanding module:

1. **Training data**
   - The knowledge injection training JSONLs (e.g., `Lysendria.jsonl`, `Aurelius_Nyxella.jsonl`) are provided under `data/knowledge`.
   - In `data/dataset_info.py`, ensure each character / pair entry under `DATASET_INFO["vlm_sft"]` has:
     - `data_dir`: folder containing rendered images for that character/pair.
     - `jsonl_path`: path to the corresponding `data/knowledge/*.jsonl`.
2. **Benchmark JSONL**
   - Evaluation JSONLs for knowledge transfer are in `benchmark/test_knowledge` (e.g., `Aurelius.jsonl`, `Aurelius_Nyxella.jsonl`, etc.).
3. **Run training**

```bash
bash train_knowledge.sh   # default: Aurelius (see --dataset_config_file in the script)
```

You can switch to other characters or pairwise knowledge injection by changing `--dataset_config_file` in `train_knowledge.sh` to the desired YAML under `data/configs/Knowledge/Forward` or `data/configs/Knowledge/Inverse`.

### 2\. Weight Completion & Conversion

Similar to the reasoning section, process the checkpoint after training:

```bash
python tool/CKPT_Transfer.py \
  --src_dir path/to/checkpoints/BAGEL-7B-MoT \        # input: original checkpoint
  --target_dirs path/to/results/checkpoints-xxx       # your checkpoint_dir
```

### 3\. Inference

Run inference on the knowledge transfer benchmarks using the **same `batch_run_eval.sh` pipeline** as in the reasoning section:

1. Edit `MODEL_PATH`, `OUTPUT_DIR`, and `FILES` in `batch_run_eval.sh` so that:
   - `MODEL_PATH` points to your converted knowledge-augmented checkpoint.
   - `FILES` contains absolute paths to JSONLs under `benchmark/test_knowledge` (e.g., `Aurelius.jsonl`, `Lysendria_Kaelorix.jsonl`, etc.).
2. Execute:

```bash
bash batch_run_eval.sh
```

Images will be saved under `OUTPUT_DIR/test/{normal,think}/{jsonl_stem}/` (e.g., `.../test/normal/Aurelius/`).

### 4\. Evaluation

Using the same vLLM setup as the Reasoning section:

**Step 1: Launch vLLM Server (if not already running)**

```bash
vllm serve path/to/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16
```

**Step 2: Run Evaluation Script**

Example command:

```bash
python eval/eval_knowledge.py \
  --jsonl benchmark/test_knowledge/Aurelius.jsonl \
  --image-dir /abs/path/to/inference_results/test/normal/Aurelius \
  --model path/to/Qwen2.5-VL-7B-Instruct \
  --overwrite
```

The script will:

- Apply the strict **person / flower / fruit** captioning and evaluation prompts.
- Produce `evaluation_results.csv` and `evaluation_results.log` in `image-dir`.

For additional usage details (e.g., JSONL format, multi-threading), please refer to the docstring and comments at the top of `eval/eval_knowledge.py`.

-----

## 🙏 Acknowledgement

This codebase is built upon [BAGEL](https://github.com/ByteDance-Seed/Bagel). We thank the authors for their great work and contribution to the community.

## 📧 Contact

If you have any questions, feel free to contact **Yuwei Niu** at [niuyuwei04@gmail.com](mailto:niuyuwei04@gmail.com).

For issues related to the BAGEL codebase, please refer to [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel).

## Citation

If you find our paper and code useful in your research, please consider giving us a star :star: and citing our work :pencil: :)
```
@article{niu2025doesunderstandinginformgeneration,
      title={Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward}, 
      author={Yuwei Niu and Weiyang Jin and Jiaqi Liao and Chaoran Feng and Peng Jin and Bin Lin and Zongjian Li and Bin Zhu and Weihao Yu and Li Yuan},
      journal={arXiv preprint arXiv:2511.20561},
      year={2025}
}
```
