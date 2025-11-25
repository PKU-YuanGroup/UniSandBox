#!/usr/bin/env python3
"""
Simple Knowledge Image Evaluation Script

Usage:
    python3 eval_knowledge.py --jsonl DATASET_JSONL --image-dir IMAGE_DIR [--model MODEL_PATH] [--max-workers N] [--overwrite]

This script:
1. Loads a JSONL file with items (fields: ID, Question, Answer).
2. For each ID, loads image <ID>.png from IMAGE_DIR.
3. Two-stage evaluation with vLLM:
   a) Generates a short caption following strict rules (person attributes / flower / fruit / Reject).
   b) Evaluates semantic match between caption and ground truth.
4. Writes CSV and log files into IMAGE_DIR.
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

# Prompts copied verbatim – DO NOT MODIFY ANY CHARACTER
CAPTION_PROMPT = (
"You are a hyper-precise and cautious vision assistant.\n"
"Task: Produce a single-line caption for the given image.\n\n"
"Rules:\n"
"1) If any person is visible, output strictly in this format:\n"
"   Person: <skin_tone_or_ethnicity>; <hair_color>; <age_group: kid|mid-age|old>; <gender: male|female|unclear>\n"
"   - **Crucial:** The <skin_tone_or_ethnicity> slot MUST be one of these three options exactly: African/Indigenous, Caucasian, East Asian.\n"
"   - If the person does not clearly fit one of these categories or you are uncertain, output 'unclear' for that slot.\n"
"   - If uncertain about other attributes (hair, age, gender), output 'unclear' for that specific slot.\n\n"
"2) If no person is visible, you MUST follow these steps:\n"
"   a. First, determine if the primary subject is a *single type* of flower OR a *single type* of fruit.\n"
"   b. If it IS a flower, you MUST output its *specific* type/species in this format: Flower: <specific_type_name> (e.g., 'Flower: rose', 'Flower: tulip').\n"
"   c. If it IS a fruit, you MUST output its *specific* type/species in this format: Fruit: <specific_type_name> (e.g., 'Fruit: apple', 'Fruit: banana').\n"
"   d. **ABSOLUTE RULE:** General categories like 'Flower: flower' or 'Fruit: fruit' are FORBIDDEN.\n\n"
"3) REJECTION: You MUST output exactly 'Reject' if ANY of the following are true:\n"
"   a. The image contains no person, AND the primary subject is **NOT** a flower or a fruit (e.g., it is a car, dog, building, etc.).\n"
"   b. (Per Rule 2) The image IS a flower or fruit, but you cannot confidently identify its *specific type* (e.g., you can only tell it's a 'fruit', not an 'apple'). In this case, you MUST output 'Reject'.\n"
"   c. The image is distorted, unrealistic, surreal (e.g., face on fruit), or generally ambiguous.\n"
"   d. The image contains multiple different types of flowers (e.g., a bouquet of roses and tulips) or multiple different types of fruits (e.g., a bowl with apples and bananas).\n"
"   e. (Per Rule 1) The person's ethnicity is visible but does not fit the three required categories.\n\n"
"4) Output must be ONE line with no extra words, no explanations.\n"
)

EVAL_PROMPT_TEMPLATE = (
"You are a careful evaluator. Determine if a generated caption and a ground truth (GT) match semantically.\n\n"
"Strict rules:\n"
"1) If the generated caption is exactly 'Reject', it is automatically incorrect. Respond with 'Score: NO'.\n"
"2) If the generated caption starts with \"Person:\", it contains 4 slots: skin/ethnicity; hair color; age group (kid|mid-age|old); gender (male|female|unclear).\n   - The GT must be a sentence describing a person (e.g., 'An elderly African/Indigenous woman with black hair.').\n   - Compare each slot against the GT description. Allow common synonyms (e.g., blond=blonde; elderly/senior=old).\n   - The generated ethnicity MUST be one of [African/Indigenous, Caucasian, East Asian] or 'unclear'.\n"
"3) If the generated caption starts with \"Flower:\" or \"Fruit:\", compare the specific object type with the GT.\n   - The GT must be a specific type (e.g., 'carnation', 'apple').\n   - **Crucial:** The generated caption *format* (Flower:/Fruit:) AND the *specific type* must BOTH match the GT.\n   - **Singular/Plural forms ARE a match.** (e.g., 'apple' matches 'apples'; 'peach' matches 'peaches').\n   - Only semantic equivalents are a match (e.g., cup~mug).\n"
"   - Example 1 (Match): Generated 'Flower: rose' and GT 'rose' is 'Score: YES'.\n   - Example 2 (Match): Generated 'Fruit: apple' and GT 'apple' is 'Score: YES'.\n   - **Example 3 (Match - Plural): Generated 'Fruit: apple' and GT 'apples' is 'Score: YES'.**\n   - **Example 4 (Match - Plural): Generated 'Fruit: peach' and GT 'peaches' is 'Score: YES'.**\n   - Example 5 (Mismatch - Wrong Type): Generated 'Flower: rose' and GT 'carnation' is 'Score: NO'.\n   - Example 6 (Mismatch - General Term): Generated 'Flower: flower' and GT 'carnation' is 'Score: NO'.\n   - **Example 7 (Mismatch - Wrong Category): Generated 'Fruit: rose' and GT 'rose' is 'Score: NO'.**\n   - **Example 8 (Mismatch - Wrong Category): Generated 'Flower: apple' and GT 'apple' is 'Score: NO'.**\n"
"4) If formats fundamentally differ (e.g., generated caption starts with 'Person:' while GT is 'apple'), respond with 'Score: NO'.\n\n"
"Input:\nGenerated: {caption}\nGroundTruth: {gt}\n\nOutput format (MUST be exact):\nScore: YES or Score: NO"
)

class KnowledgeEvaluator:
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1", model_path: str = "/mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct", max_workers: int = 32, max_retries: int = 3, overwrite: bool = False):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_path = model_path
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.file_lock = threading.Lock()
        self.api_lock = threading.Semaphore(max_workers)

    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def encode_image(img_path: str) -> str:
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _generate_caption(self, img_path: str) -> Optional[Tuple[str, str]]:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print(f"Retry {attempt} caption for {os.path.basename(img_path)}")
                b64 = self.encode_image(img_path)
                url = f"data:image;base64,{b64}"
                with self.api_lock:
                    resp = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{
                            'role': 'user',
                            'content': [
                                {'type': 'image_url', 'image_url': {'url': url}},
                                {'type': 'text', 'text': CAPTION_PROMPT},
                            ],
                        }],
                        max_tokens=2048,
                        temperature=0,
                    )
                if not resp.choices:
                    continue
                text = (resp.choices[0].message.content or '').strip()
                if text == '':
                    continue
                first_line = text.splitlines()[0].strip()
                return first_line, text
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
        print(f"Failed caption: {last_err}")
        return None

    def _evaluate_match(self, caption: str, gt: str) -> Optional[Tuple[bool, str]]:
        prompt = EVAL_PROMPT_TEMPLATE.format(caption=caption, gt=gt)
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print("Retry evaluating match")
                with self.api_lock:
                    resp = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{'role': 'user', 'content': prompt}],
                        max_tokens=1024,
                        temperature=0,
                    )
                if not resp.choices:
                    continue
                text = (resp.choices[0].message.content or '').strip()
                m = re.search(r'Score:\s*(YES|NO)', text, re.IGNORECASE)
                if m:
                    return (m.group(1).upper() == 'YES'), text
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
        print(f"Failed eval: {last_err}")
        return None

    def process_item(self, item: Dict, img_dir: str) -> Dict:
        id_val = int(item['ID']) if not isinstance(item['ID'], int) else item['ID']
        img_path = os.path.join(img_dir, f"{id_val}.png")
        row = {
            'ID': id_val,
            'Question': item.get('Question', ''),
            'Answer': item.get('Answer', ''),
            'Image_Path': img_path,
        }
        if os.path.exists(img_path):
            try:
                cap_res = self._generate_caption(img_path)
                if cap_res is None:
                    raise RuntimeError('caption failed')
                caption, caption_full = cap_res
                if caption == 'Reject':
                    is_match = False
                    eval_full = 'Caption=Reject => automatic NO'
                else:
                    eval_res = self._evaluate_match(caption, row['Answer'])
                    if eval_res is None:
                        raise RuntimeError('eval failed')
                    is_match, eval_full = eval_res
                row['Caption'] = caption
                row['Result'] = 'YES' if is_match else 'NO'
                row['Score'] = 1 if is_match else 0
                row['Analysis'] = (
                    f"Phase 1 - Caption:\n{caption_full}\n\n"
                    f"Phase 2 - Evaluation:\nGenerated: {caption}\nGT: {row['Answer']}\n{eval_full}"
                )
            except Exception as e:
                row.update({'Caption': '', 'Result': 'ERROR', 'Score': 0, 'Analysis': f'Error: {e}'})
        else:
            row.update({'Caption': '', 'Result': 'IMAGE_NOT_FOUND', 'Score': 0, 'Analysis': 'Image file not found'})
        return row

    def write_row(self, csv_path: str, headers: List[str], row: Dict):
        with self.file_lock:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=headers)
                if write_header:
                    w.writeheader()
                w.writerow(row)

    def evaluate(self, jsonl_path: str, img_dir: str):
        data = self.load_jsonl(jsonl_path)
        csv_path = os.path.join(img_dir, 'evaluation_results.csv')
        log_path = os.path.join(img_dir, 'evaluation_results.log')
        headers = ['ID', 'Question', 'Answer', 'Image_Path', 'Caption', 'Result', 'Score', 'Analysis']
        if self.overwrite and os.path.exists(csv_path):
            os.remove(csv_path)
        if self.overwrite and os.path.exists(log_path):
            os.remove(log_path)
        total = len(data)
        processed = correct = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.process_item, item, img_dir): item['ID'] for item in data}
            for fut in as_completed(futs):
                id_val = futs[fut]
                try:
                    row = fut.result()
                    self.write_row(csv_path, headers, row)
                    processed += 1
                    if row['Score'] == 1:
                        correct += 1
                    print(f"ID {id_val} done ({processed}/{total})")
                except TimeoutError:
                    print(f"ID {id_val} timeout")
                except Exception as e:
                    print(f"ID {id_val} failed: {e}")
        acc = correct / processed if processed else 0
        summary = [
            '='*60,
            'Evaluation finished!',
            f'Total: {processed}',
            f'Correct: {correct}',
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
    p = argparse.ArgumentParser(description='Simple Knowledge image evaluation')
    p.add_argument('--jsonl', required=True, help='Path to dataset JSONL')
    p.add_argument('--image-dir', required=True, help='Directory with images')
    p.add_argument('--model', default='/mnt/data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct', help='vLLM model path')
    p.add_argument('--max-workers', type=int, default=32, help='Thread pool size')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing CSV/log')
    args = p.parse_args()

    evaluator = KnowledgeEvaluator(model_path=args.model, max_workers=args.max_workers, overwrite=args.overwrite)
    evaluator.evaluate(args.jsonl, args.image_dir)

if __name__ == '__main__':
    main()
