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
OUTPUT_JSONL = "./mapping1.jsonl"
TARGET_COUNT = 64000
BATCH_SIZE = 128 
# ========================

def build_prompt():
    return """Data Generation Prompt for One-Step Symbol Mapping
Please generate **50 pairs** of one-step symbol mapping data. Each pair should consist of two prompts: `Question_A` and `Question_B`. Each mapping rule involves two symbols, and each symbol represents a common object. Each pair of prompts must clearly state the symbol mapping rule and generate two prompts: `Question_A` and `Question_B`. The answers should correspond to the objects represented by the symbols in the rules.

In this version:
- `Question_A` will generate the object represented by the **first** symbol.
- `Question_B` will generate the object represented by the **second** symbol.

Each `Question_A` and `Question_B` must have a unique `ID`, and each prompt should have an `Answer` field. The answer should be the object represented by the corresponding symbol.

Output Format:
The output should be in **JSONL** format, where each entry contains the following fields:
- **ID**: A unique identifier, starting from 1, with the same `ID` for both `Question_A` and `Question_B`.
- **Question_A**: The prompt that contains the symbol mapping rule and generates the object represented by the first symbol.
- **Question_B**: The prompt that contains the symbol mapping rule and generates the object represented by the second symbol.
- **Answer**: The corresponding answer for the prompt, which should be the object that the symbol represents.
Example Format:
{"ID": "1", "Question_A": "Rule 1: The symbol @ represents apples. Rule 2: The symbol * represents bananas. Please generate the fruit represented by the symbol @.", "Answer": "Apples"}
{"ID": "1", "Question_B": "Rule 1: The symbol @ represents apples. Rule 2: The symbol * represents bananas. Please generate the fruit represented by the symbol *.", "Answer": "Bananas"}
Requirements:
1. Symbols: Symbols can be any common symbols, including letters (e.g., a, b, c, A, B, C), numbers (e.g., 1, 2, 3, 5), punctuation marks (e.g., @, #, *, &, $, %, ^, etc.), and other characters. Symbols should not be limited to numbers or special characters but can include any alphanumeric or symbolic character.
2. Object Mapping: Each pair of data should map the symbols to common objects such as pencils, chairs, phones, refrigerators, notebooks, etc. The objects can be everyday items, not just fruits.
3. Answer Consistency: Ensure that the Question_A and Question_B are clearly structured, and that the Answer for each corresponds to the symbol mapping rule.
4. Unique IDs: Each pair of Question_A and Question_B should share the same ID.
5. Object Diversity: The objects in the answers should be varied and can include common items like fruits, stationery, household items, animals, etc.
Example Data:
{"ID": "1", "Question_A": "Rule 1: The symbol @ represents apples. Rule 2: The symbol * represents bananas. Please generate the fruit represented by the symbol @.", "Answer": "Apples"}
{"ID": "1", "Question_B": "Rule 1: The symbol @ represents apples. Rule 2: The symbol * represents bananas. Please generate the fruit represented by the symbol *.", "Answer": "Bananas"}
{"ID": "2", "Question_A": "Rule 1: The symbol # represents pencils. Rule 2: The symbol $ represents notebooks. Please generate the object represented by the symbol #.", "Answer": "Pencils"}
{"ID": "2", "Question_B": "Rule 1: The symbol # represents pencils. Rule 2: The symbol $ represents notebooks. Please generate the object represented by the symbol $.", "Answer": "Notebooks"}
{"ID": "3", "Question_A": "Rule 1: The symbol % represents chairs. Rule 2: The symbol ^ represents tables. Please generate the object represented by the symbol %.", "Answer": "Chairs"}
{"ID": "3", "Question_B": "Rule 1: The symbol % represents chairs. Rule 2: The symbol ^ represents tables. Please generate the object represented by the symbol ^.", "Answer": "Tables"}
{"ID": "4", "Question_A": "Rule 1: The symbol * represents apples. Rule 2: The symbol @ represents oranges. Please generate the fruit represented by the symbol *.", "Answer": "Apples"}
{"ID": "4", "Question_B": "Rule 1: The symbol * represents apples. Rule 2: The symbol @ represents oranges. Please generate the fruit represented by the symbol @.", "Answer": "Oranges"}
{"ID": "5", "Question_A": "Rule 1: The symbol $ represents pencils. Rule 2: The symbol # represents erasers. Please generate the object represented by the symbol $.", "Answer": "Pencils"}
{"ID": "5", "Question_B": "Rule 1: The symbol $ represents pencils. Rule 2: The symbol # represents erasers. Please generate the object represented by the symbol #.", "Answer": "Erasers"}
Guidelines:
1. Symbols: Feel free to use any common symbols, including alphanumeric characters (e.g., a, b, c, A, B, C), numbers (e.g., 1, 2, 3), and special characters (e.g., @, #, *, &, $, %, ^, etc.).
2. Objects: Objects should be common, everyday items such as stationery, fruits, animals, etc.
3. Output: Ensure the JSONL format is strictly followed, and each data entry is separate.
4. Answer Consistency: Each symbol must be mapped to an appropriate common object, and the Answer should reflect this mapping.
Please generate the data based on these instructions.

"""

def parse_response(response_text):
    lines = response_text.strip().split('\n')
    valid_items = []
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('{') and line.endswith('}'):
            try:
                item = json.loads(line)
                if ("Question_A" in item or "Question_B" in item) and "Answer" in item:
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
        
        print(f"\n--- Batch {batch_id} Raw GPT Output ---")
        print(result_text)
        print("-------------------------------------------\n")

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
                print(f"✅ Batch {result['batch_id']}: Successfully parsed and generated {result['count']} items")
            else:
                print(f"❌ Batch {result['batch_id']}: Failed or no valid items parsed")
    
    if len(all_items) > TARGET_COUNT:
        all_items = all_items[:TARGET_COUNT]
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Generation completed!")
    print(f"📊 Target count: {TARGET_COUNT}")
    print(f"📊 Actually parsed and saved: {len(all_items)}")
    print(f"💾 Save location: {OUTPUT_JSONL}")
    
    if all_items:
        print(f"\n📝 Sample parsed data (first 3):")
        for i, item in enumerate(all_items[:3]):
            if "Question_A" in item:
                print(f"  {i+1}. ID: {item.get('ID', 'N/A')}")
                print(f"     Question_A: {item['Question_A']}")
                print(f"     Answer: {item['Answer']}")
            elif "Question_B" in item:
                print(f"  {i+1}. ID: {item.get('ID', 'N/A')}")
                print(f"     Question_B: {item['Question_B']}")
                print(f"     Answer: {item['Answer']}")
            print("-" * 20)

if __name__ == "__main__":
    main()