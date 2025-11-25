#!/usr/bin/env python3
"""
Math Generation Evaluation Script

Usage:
    python3 eval_math.py --jsonl DATASET_JSONL --image-dir IMAGE_DIR [--model MODEL_PATH] [--max-workers N] [--overwrite]
    
    
    conda activate eval-vllm

vllm serve /mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct \
--port 8000 \
--host 0.0.0.0 \
--dtype bfloat16 
    python3 /mnt/data/nyw/STARS/eval/eval_math.py --jsonl /mnt/data/nyw/whyuni_data/test/math_1.jsonl --image-dir /mnt/data/nyw/eval_results/blip3o_8B/test/normal/math_1 --overwrite
     [--model MODEL_PATH] [--max-workers N] [--overwrite]

This script:
1. Loads a JSONL file with problem items (expects fields: id, Question, Answer).
2. For each item, loads the corresponding image <id>.png from IMAGE_DIR.
3. Runs a two–stage evaluation using a vLLM service:
   a) Generates a strict caption describing the visible objects.
   b) Compares the caption with the expected answer.
4. Writes a CSV result file and a log file into IMAGE_DIR.
"""

import csv
import json
import os
import threading
import time
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Tuple, Optional

from openai import OpenAI

# prompts below must not be modified – keep exactly the same strings
CAPTION_PROMPT = """You are an extremely rigorous and neutral image captioning expert. Your task is to carefully examine a given image and generate a precise, concise caption (under 20 characters) that strictly and unambiguously describes the count and type of all clearly, fully, and normally visible objects in the image.

### Instructions:

1.  **Only count a single object type:** The image must contain only one clearly identifiable type of object. If multiple distinct object types are present, the task is considered an error.
2.  **Only include objects that are:**
    * **Fully visible**
    * **Clearly identifiable**
    * **Normal in shape, scale, and orientation**
    * **Not distorted, occluded, cropped, or ambiguous**
3.  **Strict Error Handling:** If the image contains more than one type of object, or if any visible object is distorted or has its geometry compromised, the task is considered an error.
4.  **Your caption must be in the format:**
    * `[Number] [object_type_plural]` e.g., "3 apples", "2 chairs"
5.  **If the conditions in rules 1-3 are not met, you must output 'Error'. Do not provide any other explanations.**

### Output Format:

Caption: [your output here]"""

EVALUATION_PROMPT_TEMPLATE = """You are a rigorous and fair caption evaluator. Your task is to compare a generated image caption with an expected answer, and determine whether they match **perfectly and exclusively in count and object meaning**.

### **Evaluation Rules**

1.  The caption must match the expected answer in:
    * Count: The number of each object type must be exactly the same.
    * Object Type: The object types in both descriptions must be exactly the same.

2.  If any object type in the caption does not match the expected answer, or if the count is wrong, or if there are extra object types, return:
    **Score: NO**

3.  If and only if the object types and counts in the caption exactly match the expected answer, return:
    **Score: YES**

---
### **Input**

**Generated Caption:** {caption}
**Expected Answer:** {expected_answer}

---
### **Output Format**

**Score: [YES or NO]**"""

class MathEvaluator:
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1", model_path: str = "/mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct", max_workers: int = 32, max_retries: int = 3, overwrite: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_path = model_path
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.file_lock = threading.Lock()
        self.api_call_lock = threading.Semaphore(max_workers)

    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        data: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _generate_image_caption(self, image_path: str) -> Optional[Tuple[str, str]]:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Retry {attempt} generating caption for {os.path.basename(image_path)}")
                base64_img = self.encode_image(image_path)
                b64_url = f"data:image;base64,{base64_img}"
                with self.api_call_lock:
                    response = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": b64_url}},
                                {"type": "text", "text": CAPTION_PROMPT},
                            ],
                        }],
                        max_tokens=10240,
                        temperature=0,
                    )
                if not response.choices:
                    continue
                message = response.choices[0].message
                result_text = getattr(message, "content", "")
                if not result_text:
                    continue
                if result_text.strip().upper() == "ERROR":
                    return "Error", result_text
                return result_text.strip(), result_text
            except Exception as e:
                last_error = e
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
        print(f"Failed to generate caption: {last_error}")
        return None

    def _evaluate_caption_match(self, caption: str, expected_answer: str) -> Optional[Tuple[bool, str]]:
        last_error = None
        prompt = EVALUATION_PROMPT_TEMPLATE.format(caption=caption, expected_answer=expected_answer)
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print("Retry evaluating caption match")
                with self.api_call_lock:
                    response = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10240,
                        temperature=0,
                    )
                if not response.choices:
                    continue
                result_text = getattr(response.choices[0].message, "content", "")
                m = re.search(r"Score:\s*(YES|NO)", result_text, re.IGNORECASE)
                if m:
                    is_match = m.group(1).upper() == "YES"
                    return is_match, result_text
            except Exception as e:
                last_error = e
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
        print(f"Failed to evaluate caption match: {last_error}")
        return None

    def analyze_image(self, image_path: str, expected_answer: str) -> Tuple[bool, str]:
        caption_result = self._generate_image_caption(image_path)
        if caption_result is None:
            return False, "Error: Failed to generate image caption"
        caption, caption_full = caption_result
        if caption.upper().strip() == "ERROR":
            return False, f"Phase 1 - Image Captioning:\nModel returned 'Error', skipping Phase 2."
        evaluation_result = self._evaluate_caption_match(caption, expected_answer)
        if evaluation_result is None:
            return False, f"Error: Failed to evaluate caption match\nGenerated Caption: {caption}"
        is_match, eval_text = evaluation_result
        combined = (
            f"Phase 1 - Image Captioning:\n{caption_full}\n\n"
            f"Phase 2 - Caption Evaluation:\nGenerated Caption: {caption}\nExpected Answer: {expected_answer}\n{eval_text}"
        )
        return is_match, combined

    def process_item(self, item: Dict, image_dir: str) -> Dict:
        id_val = item["id"]
        image_path = os.path.join(image_dir, f"{id_val}.png")
        row = {
            "ID": id_val,
            "Question": item.get("Question", ""),
            "Answer": item.get("Answer", ""),
            "Image_Path": image_path,
        }
        if os.path.exists(image_path):
            try:
                is_match, analysis = self.analyze_image(image_path, row["Answer"])
                row["Result"] = "YES" if is_match else "NO"
                row["Score"] = 1 if is_match else 0
                row["Analysis"] = analysis
            except Exception as e:
                row["Result"] = "ERROR"
                row["Score"] = 0
                row["Analysis"] = f"Error: {e}"
        else:
            row["Result"] = "IMAGE_NOT_FOUND"
            row["Score"] = 0
            row["Analysis"] = "Image file not found"
        return row

    def write_csv_row(self, csv_path: str, headers: List[str], row: Dict):
        with self.file_lock:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    def evaluate_dataset(self, jsonl_path: str, image_dir: str):
        data = self.load_jsonl(jsonl_path)
        csv_path = os.path.join(image_dir, "evaluation_results.csv")
        log_path = os.path.join(image_dir, "evaluation_results.log")
        headers = ["ID", "Question", "Answer", "Image_Path", "Result", "Score", "Analysis"]

        if self.overwrite and os.path.exists(csv_path):
            os.remove(csv_path)
        if self.overwrite and os.path.exists(log_path):
            os.remove(log_path)

        start_time = time.time()
        processed = 0
        correct = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_item, item, image_dir): item for item in data}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    row = future.result()
                    self.write_csv_row(csv_path, headers, row)
                    processed += 1
                    if row["Score"] == 1:
                        correct += 1
                    print(f"ID {item['id']} done ({processed}/{len(data)})")
                except TimeoutError:
                    print(f"ID {item['id']} timeout")
                except Exception as e:
                    print(f"ID {item['id']} failed: {e}")

        accuracy = correct / processed if processed else 0
        duration = time.time() - start_time
        summary_lines = [
            "=" * 60,
            "Evaluation finished!",
            f"Total: {processed}",
            f"Correct: {correct}",
            f"Accuracy: {accuracy:.2%}",
            f"Duration: {duration:.2f}s",
            f"CSV: {csv_path}",
            "=" * 60,
        ]
        with open(log_path, "w", encoding="utf-8") as f:
            for line in summary_lines:
                f.write(line + "\n")
        print("\n".join(summary_lines))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Math evaluation")
    parser.add_argument("--jsonl", required=True, help="Path to dataset JSONL file")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV/log files")
    parser.add_argument("--max-workers", type=int, default=32, help="Thread pool size")
    parser.add_argument("--model", type=str, default="/mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct", help="Path to vLLM model")
    args = parser.parse_args()

    evaluator = MathEvaluator(model_path=args.model, max_workers=args.max_workers, overwrite=args.overwrite)
    evaluator.evaluate_dataset(args.jsonl, args.image_dir)

if __name__ == "__main__":
    main()
