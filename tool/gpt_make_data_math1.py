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
OUTPUT_JSONL = "./math_1.jsonl"
TARGET_COUNT = 64000
BATCH_SIZE = 128 
# ========================

def build_prompt():
    return """
Please generate 50 prompts in the following JSONL format:
{"Question": "Produce a number of pencils equal to the result of 2 * 2.","Answer": "4 pencils."}
Requirements:
1. Each Question must explicitly instruct the user to perform a basic arithmetic operation (addition, subtraction, multiplication, or division), with the final result strictly between 1 and 4 (inclusive).
2. Use a diverse and natural set of expressions, such as:
  - “Generate as many [objects] as the result of [expression].”
  - “Produce a number of [objects] equal to the result of [expression].”
  - “Show the number of [objects] that matches the outcome of [expression].”
  - “Create the same quantity of [objects] as [expression] equals.”
  - “Provide the same number of [objects] as calculated by [expression].”
3. Replace [expression] with a valid arithmetic expression (e.g., 3 - 1, 2 + 2, 4 / 2) that evaluates to 1, 2, 3, or 4.
4. Use a wide variety of common objects (not just fruits). Include animals, toys, stationery, kitchen items, etc. Do not use rare or unusual items.
5. In the Answer, the object name must be grammatically correct and match the number, e.g.,:
  - "1 eraser"
  - "2 oranges"
  - "3 kittens"
  - "4 spoons"
6. Output Format:
  - Please return the 1000 generated items as individual JSON objects, one after another, not wrapped in a list or array.
  - Do not include additional text, titles, or explanations.
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