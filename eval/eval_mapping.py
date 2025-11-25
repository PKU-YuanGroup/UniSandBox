#!/usr/bin/env python3
"""
Mapping Generation Evaluation Script

Usage:
    python3 eval_mapping.py --jsonl DATASET_JSONL --image-dir IMAGE_DIR [--model MODEL_PATH] [--max-workers N] [--overwrite]

This script:
1. Loads a JSONL file with items A/B per ID (fields: ID, Question_A / Question_B, Answer).
2. For each ID, loads images <ID>_A.png and <ID>_B.png from IMAGE_DIR.
3. Runs a two–stage evaluation per image using a vLLM service:
   a) Identifies the primary object type.
   b) Compares the identified object with the expected answer.
4. Both A and B must be correct for the ID to score 1.
5. Writes CSV and log files into IMAGE_DIR.
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

# Prompts below are copied verbatim — DO NOT MODIFY
IDENTIFICATION_PROMPT = """You are an extremely rigorous and neutral image analysis expert. Your task is to carefully examine a given image and identify the primary object type that is clearly, fully, and normally visible in the image.

### Instructions:

1.  **Only identify objects that are:**
    * **Fully visible**
    * **Clearly identifiable**
    * **Normal in shape, scale, and orientation**
    * **Not distorted, occluded, cropped, or ambiguous**

2.  **Strict Error Handling:** Strict Error Handling: The task is considered an error if any of the following conditions are met:
    * The image contains more than one distinct type of object.
    * Any visible object is distorted or has its geometry compromised.
    * *The image contains any letters (e.g., A, B, C), symbols (e.g., punctuation marks, mathematical operators like +, -, , #, $), or numbers (e.g., 1, 2, 3).*

3.  **Zero Tolerance for Text and Symbols:** This is a **critical rule**. The task is an automatic error if the image contains:
    * **Any letters (e.g., A, B, C)**
    * **Any symbols (e.g., punctuation, mathematical operators like +, -, =, #, $)**
    * **Any numbers (e.g., 1, 2, 3)**

4.  **Your identification must be in the format:**
    * `[object_type]` e.g., "apple", "chair", "car"
    * Use the most specific and accurate term for the object type

4.  **If the conditions in rules 1-3 are not met, you must output 'Error'. Do not provide any other explanations.**

### Output Format:

Object: [your output here]"""

EVALUATION_PROMPT_TEMPLATE = """You are a rigorous and fair object evaluator. Your task is to compare an identified object type with an expected answer, and determine whether they match **semantically**.

### **Evaluation Rules**

1.  The identified object must match the expected answer in meaning:
    * Object types may be considered a match if they are semantically equivalent (e.g., "car" and "automobile", "cup" and "mug")
    * Consider synonyms, alternative names, and related terms

2.  If the identified object type does not semantically match the expected answer, return:
    **Score: NO**

3.  If and only if the identified object type semantically matches the expected answer, return:
    **Score: YES**

---
### **Input**

**Identified Object:** {identified_object}
**Expected Answer:** {expected_answer}

---
### **Output Format**

**Score: [YES or NO]**"""

class MappingEvaluator:
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

    # --- stage 1
    def _identify_object(self, image_path: str) -> Optional[Tuple[str, str]]:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Retry {attempt} identifying object for {os.path.basename(image_path)}")
                b64 = self.encode_image(image_path)
                url = f"data:image;base64,{b64}"
                with self.api_call_lock:
                    resp = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                                {"type": "text", "text": IDENTIFICATION_PROMPT},
                            ],
                        }],
                        max_tokens=10240,
                        temperature=0,
                    )
                if not resp.choices:
                    continue
                text = getattr(resp.choices[0].message, "content", "")
                if not text:
                    continue
                if text.strip().upper() == "ERROR":
                    return "Error", text
                m = re.search(r"Object:\s*(.+)", text.strip())
                obj = m.group(1).strip() if m else text.strip()
                return obj, text
            except Exception as e:
                last_error = e
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
        print(f"Failed to identify object: {last_error}")
        return None

    # --- stage 2
    def _evaluate_match(self, identified: str, expected: str) -> Optional[Tuple[bool, str]]:
        prompt = EVALUATION_PROMPT_TEMPLATE.format(identified_object=identified, expected_answer=expected)
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print("Retry evaluating match")
                with self.api_call_lock:
                    resp = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10240,
                        temperature=0,
                    )
                if not resp.choices:
                    continue
                text = getattr(resp.choices[0].message, "content", "")
                m = re.search(r"Score:\s*(YES|NO)", text, re.IGNORECASE)
                if m:
                    is_match = m.group(1).upper() == "YES"
                    return is_match, text
            except Exception as e:
                last_error = e
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
        print(f"Failed to evaluate match: {last_error}")
        return None

    def analyze_image(self, image_path: str, expected_answer: str) -> Tuple[bool, str]:
        ident_res = self._identify_object(image_path)
        if ident_res is None:
            return False, "Error: Failed to identify object"
        obj, obj_full = ident_res
        if obj.upper().strip() == "ERROR":
            return False, f"Phase 1 - Object Identification:\nModel returned 'Error', skipping Phase 2."
        eval_res = self._evaluate_match(obj, expected_answer)
        if eval_res is None:
            return False, f"Error: Failed to evaluate match\nIdentified Object: {obj}"
        is_match, eval_text = eval_res
        combined = (
            f"Phase 1 - Object Identification:\n{obj_full}\n\n"
            f"Phase 2 - Object Evaluation:\nIdentified Object: {obj}\nExpected Answer: {expected_answer}\n{eval_text}"
        )
        return is_match, combined

    # process A/B per ID
    def process_id(self, id_val: int, group: Dict, image_dir: str) -> Dict:
        row = {'ID': id_val}
        a_ok = b_ok = False
        for suffix in ['A', 'B']:
            if suffix not in group:
                continue
            item = group[suffix]
            q_key = f'Question_{suffix}'
            answer = item['Answer']
            img_path = os.path.join(image_dir, f"{id_val}_{suffix}.png")
            row[q_key] = item[q_key]
            row[f'Answer_{suffix}'] = answer
            row[f'Image_{suffix}_Path'] = img_path
            if os.path.exists(img_path):
                try:
                    is_match, analysis = self.analyze_image(img_path, answer)
                    row[f'{suffix}_Result'] = 'YES' if is_match else 'NO'
                    row[f'{suffix}_Score'] = 1 if is_match else 0
                    row[f'{suffix}_Analysis'] = analysis
                    if suffix == 'A':
                        a_ok = is_match
                    else:
                        b_ok = is_match
                except Exception as e:
                    row[f'{suffix}_Result'] = 'ERROR'
                    row[f'{suffix}_Score'] = 0
                    row[f'{suffix}_Analysis'] = f"Error: {e}"
            else:
                row[f'{suffix}_Result'] = 'IMAGE_NOT_FOUND'
                row[f'{suffix}_Score'] = 0
                row[f'{suffix}_Analysis'] = 'Image file not found'
        both = a_ok and b_ok
        row['Both_Correct'] = both
        row['Final_Score'] = 1 if both else 0
        return row

    def write_row(self, csv_path: str, headers: List[str], row: Dict):
        with self.file_lock:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=headers)
                if write_header:
                    w.writeheader()
                w.writerow(row)

    def evaluate_dataset(self, jsonl_path: str, image_dir: str):
        data = self.load_jsonl(jsonl_path)
        # group by ID
        groups: Dict[int, Dict] = {}
        for item in data:
            id_val = item['ID'] if isinstance(item['ID'], int) else int(item['ID'])
            if id_val not in groups:
                groups[id_val] = {}
            if 'Question_A' in item:
                groups[id_val]['A'] = item
            elif 'Question_B' in item:
                groups[id_val]['B'] = item
        csv_path = os.path.join(image_dir, 'evaluation_results.csv')
        log_path = os.path.join(image_dir, 'evaluation_results.log')
        headers = ['ID', 'Question_A', 'Answer_A', 'Image_A_Path', 'A_Result', 'A_Score', 'A_Analysis',
                   'Question_B', 'Answer_B', 'Image_B_Path', 'B_Result', 'B_Score', 'B_Analysis',
                   'Both_Correct', 'Final_Score']
        if self.overwrite and os.path.exists(csv_path):
            os.remove(csv_path)
        if self.overwrite and os.path.exists(log_path):
            os.remove(log_path)
        total_ids = len(groups)
        processed = correct = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.process_id, id_val, grp, image_dir): id_val for id_val, grp in groups.items()}
            for fut in as_completed(futs):
                id_val = futs[fut]
                try:
                    row = fut.result()
                    self.write_row(csv_path, headers, row)
                    processed += 1
                    if row['Final_Score'] == 1:
                        correct += 1
                    print(f"ID {id_val} done ({processed}/{total_ids})")
                except TimeoutError:
                    print(f"ID {id_val} timeout")
                except Exception as e:
                    print(f"ID {id_val} failed: {e}")
        acc = correct / processed if processed else 0
        duration = time.time() - 0  # start not stored but fine
        summary = [
            '='*60,
            'Evaluation finished!',
            f'Total IDs: {processed}',
            f'Correct IDs: {correct}',
            f'Accuracy: {acc:.2%}',
            f'CSV: {csv_path}',
            '='*60,
        ]
        with open(log_path, 'w', encoding='utf-8') as f:
            for line in summary:
                f.write(line + '\n')
        print('\n'.join(summary))


def main():
    import argparse
    p = argparse.ArgumentParser(description='Simple Mapping image evaluation')
    p.add_argument('--jsonl', required=True, help='Path to dataset JSONL')
    p.add_argument('--image-dir', required=True, help='Directory with images')
    p.add_argument('--model', default='/mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct', help='vLLM model path')
    p.add_argument('--max-workers', type=int, default=32, help='Thread pool size')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing CSV/log')
    args = p.parse_args()

    evaluator = MappingEvaluator(model_path=args.model, max_workers=args.max_workers, overwrite=args.overwrite)
    evaluator.evaluate_dataset(args.jsonl, args.image_dir)

if __name__ == '__main__':
    main()
