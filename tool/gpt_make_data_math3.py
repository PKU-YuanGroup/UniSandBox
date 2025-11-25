import json
import requests
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

API_KEY = ""
BASE_URL = ""

MODEL = "gpt-4o"
MAX_THREADS = 64
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
API_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
OUTPUT_JSONL = "./math_3.jsonl"
TARGET_COUNT = 64000
BATCH_SIZE = 128 
# ========================

def build_prompt():
    return """Please generate 50 prompts in the following JSONL format:

{"Question": "Produce a number of pencils equal to the result of √9 + 2 - 3.", "Answer": "2 pencils"}

Requirements:
1. Each Question must explicitly instruct the user to perform a **three-step arithmetic operation** (e.g., addition, subtraction, multiplication, division, square root, exponentiation, modulo, etc.), with the final result strictly between 1 and 5 (inclusive).
2. The operation must involve **three different or consecutive mathematical operations**, such as:
    - "Add 3 to the square root of 4, then subtract 1."
    - "Multiply 2 by the square root of 9, then add 1."
    - "Subtract 1 from the square root of 16, then divide by 1."
    - "Divide 12 by the square root of 4, then add 2."
    - "Calculate the result of 2 raised to the power of 2, then subtract 2, then add 3."
    - "Find the modulo of 7 by 4, then add 2, then multiply by 1."
    - "Square 3, then subtract 5, then add 2."
3. Expression Examples:
    - Simple addition/subtraction or multiplication/division combined with square roots, exponents, modulo, or other basic mathematical functions.
    - Use expressions like "(4 + 2 - 5)^0.5", "√9 + 2 - 3", "(2^2 - 2) * 2", "4 / √16 + 3", "(7 % 4 + 2) - 1", "(3^2 - 5) + 3", etc.
4. **Important Note**: "Square", "exponentiation", "square root", etc., all count as one operation in the three-step arithmetic.
5. Each operation must have a natural set of expressions:
    - "Generate as many [objects] as the result of [expression]."
    - "Produce a number of [objects] equal to the result of [expression]."
    - "Show the number of [objects] that matches the outcome of [expression]."
    - "Create the same quantity of [objects] as [expression] equals."
    - "Provide the same number of [objects] as calculated by [expression]."
6. Replace [expression] with a valid three-step arithmetic expression involving square roots, exponents, addition, subtraction, multiplication, division, or modulo.
7. Objects should be varied, such as animals, toys, stationery, kitchen items, etc. Avoid rare or unusual items.
8. Answer should match the object name grammatically and be plural or singular depending on the result, e.g.:
    - "1 pencil"
    - "2 oranges"
    - "3 kittens"
    - "4 spoons"
9. Output Format:
    - Please return the 50 generated items as individual JSON objects, one after another, not wrapped in a list or array.
    - Do not include additional text, titles, or explanations.

Example Questions and Answers:
{"Question": "Produce a number of pencils equal to the result of √9 + 2 - 3.", "Answer": "2 pencils"}
{"Question": "Show the number of pencils that matches the outcome of (2^2 - 2) * 2.", "Answer": "4 pencils"}
{"Question": "Create the same quantity of pencils as 4 / √16 + 3.", "Answer": "4 pencils"}
{"Question": "Provide the same number of pencils as (7 % 4 + 2) - 1.", "Answer": "2 pencils"}
{"Question": "Give the same quantity of pencils as (12 / √9) + 1.", "Answer": "5 pencils"}
{"Question": "Show the number of pencils that equals (3^3 / 3) - 6.", "Answer": "3 pencils"}
{"Question": "Create as many pencils as the square root of 16 minus 2, then add 3.", "Answer": "5 pencils"}
{"Question": "Provide the number of pencils equal to (6 % 4 + 1) * 1.", "Answer": "3 pencils"}
"""

def parse_response(response_text):
    lines = response_text.strip().split('\n')
    valid_items = []
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('{') and line.endswith('}'):
            try:
                item = json.loads(line)
                if "Question" in item and "Answer" in item:
                    valid_items.append(item)
            except json.JSONDecodeError:
                continue
    
    return valid_items

def call_api(batch_id):
    prompt = build_prompt()
    
    try:
        response = requests.post(
            API_ENDPOINT,
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0
            },
            timeout=120
        )
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        
        items = parse_response(result_text)
        
        return {
            "batch_id": batch_id,
            "success": True,
            "items": items,
            "count": len(items)
        }
        
    except Exception as e:
        print(f"[ERROR] Batch {batch_id} failed: {e}")
        return {
            "batch_id": batch_id,
            "success": False,
            "items": [],
            "count": 0
        }

def main():
    print(f"🚀 Starting generation of {TARGET_COUNT} prompts...")
    print(f"📊 Configuration: {MAX_THREADS} threads, temperature 1.0, {BATCH_SIZE} per batch")
    
    all_items = []
    num_batches = (TARGET_COUNT + BATCH_SIZE - 1) // BATCH_SIZE
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(call_api, i) for i in range(num_batches)]
        
        for future in tqdm(as_completed(futures), total=num_batches, desc="Generation progress"):
            result = future.result()
            if result["success"]:
                all_items.extend(result["items"])
                print(f"✅ Batch {result['batch_id']}: Successfully generated {result['count']} items")
            else:
                print(f"❌ Batch {result['batch_id']}: Failed")
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Generation completed!")
    print(f"📊 Target count: {TARGET_COUNT}")
    print(f"📊 Actually generated: {len(all_items)}")
    print(f"💾 Save location: {OUTPUT_JSONL}")
    
    if all_items:
        print(f"\n📝 Sample data:")
        for i, item in enumerate(all_items[:3]):
            print(f"  {i+1}. Question: {item['Question']}")
            print(f"     Answer: {item['Answer']}")

if __name__ == "__main__":
    main()