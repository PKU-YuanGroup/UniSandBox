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
OUTPUT_JSONL = "mapping2.jsonl"
TARGET_COUNT = 64000
BATCH_SIZE = 128 
# ========================

def build_prompt():
    return """Data Generation Prompt for Two-Step Symbol Mapping with Randomized Rule Order

Please generate 50 pairs of two-step symbol mapping data. Each pair should consist of two prompts: Question_A and Question_B. Each prompt includes a two-level symbolic mapping:
- Some symbols (e.g., numbers or punctuation) are mapped to common objects.
- Other symbols (e.g., letters) are mapped to the first-level symbols.
- The final object is retrieved by resolving the second-level symbol through both mappings.

---

Critical Instructions:

- Use Rule-based format (e.g., “Rule 1: … Rule 2: …”), but:
  - The order of the rules must be randomized in each prompt.
  - The mapping combinations (e.g., A → 1, 1 → apple) must vary — avoid fixed pairings.
  - Do not always let Rule 1 and Rule 2 define the objects, and Rule 3 and Rule 4 define the indirection — shuffle them.

---

Output Format:

Output each prompt in JSONL format, with the following fields:
- "ID": Shared identifier between Question_A and Question_B.
- "Question_A": A full prompt containing 4 shuffled rules, ending with a question: "Please generate the [object type] represented by the [letter]." (e.g., "Please generate the fruit represented by the letter 'A'." or "Please generate the device represented by the letter 'Y'.")
- "Question_B": Same rules, same order, different query symbol.
- "Answer": The correct object for that symbol (after two-step resolution).

---

Requirements:

1. Use 4 rules per prompt, all in the form: Rule x: <mapping>.
2. Each rule should be one of:
  - <symbol> represents <object>
  - <letter> represents <symbol>
3. The rule order must be shuffled randomly for every prompt.
4. The pairings (A → 1, B → 2 vs. A → 2, B → 1, etc.) should vary.
5. Symbols can be letters, numbers, or common characters (A, B, 1, 2, @, #, etc.)
6. Objects should be diverse and common (e.g., apple, dog, notebook, laptop, spoon, etc.)
7. "Question_A" must always ask for the object represented by the first letter symbol.
8. "Question_B" must always ask for the object represented by the second letter symbol.
9. Both prompts must use the same 4 rules, in the same shuffled order.

---

Example Output:

JSONL
{"ID": "1", "Question_A": "Rule 1: The number 2 stands for bananas. Rule 2: The letter 'B' refers to the number 2. Rule 3: The number 1 means apple. Rule 4: The letter 'A' refers to the number 1. Please generate the fruit represented by the letter 'A'.", "Answer": "apple"}
{"ID": "1", "Question_B": "Rule 1: The number 2 stands for bananas. Rule 2: The letter 'B' refers to the number 2. Rule 3: The number 1 means apple. Rule 4: The letter 'A' refers to the number 1. Please generate the fruit represented by the letter 'B'.", "Answer": "bananas"}

{"ID": "2", "Question_A": "Rule 1: The symbol # means keyboard. Rule 2: Letter X maps to @. Rule 3: @ means monitor. Rule 4: Letter Y maps to #. Please generate the device represented by the letter 'Y'.", "Answer": "keyboard"}
{"ID": "2", "Question_B": "Rule 1: The symbol # means keyboard. Rule 2: Letter X maps to @. Rule 3: @ means monitor. Rule 4: Letter Y maps to #. Please generate the device represented by the letter 'X'.", "Answer": "monitor"}

---
Guidelines:
- Ensure answers are valid two-step resolutions.
- Keep prompt language consistent, factual, and rule-based.
- Vary symbols and objects across examples to avoid redundancy.
- Choose familiar objects in daily life, and it's best if there's a significant difference between the two objects.
- Output must be strict JSONL, one object per line, no extra formatting or explanation.

---
Please generate the data based on these instructions.

---
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