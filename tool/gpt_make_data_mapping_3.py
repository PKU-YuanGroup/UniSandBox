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
OUTPUT_JSONL = "./mapping3.jsonl"
TARGET_COUNT = 64000
BATCH_SIZE = 128 
# ========================

def build_prompt():
    return """Data Generation Prompt for Three-Step Symbol Mapping with Randomized Rule Order

Please generate 50 pairs of three-step symbol mapping data. Each pair must consist of two prompts: Question_Aand Question_B.
Each prompt involves a three-level symbolic mapping:
- First, base symbols (e.g., numbers, punctuation) map to common objects.
- Second, another set of symbols (e.g., letters) map to the base symbols.
- Third, higher-level symbols (e.g., other letters) map to the second-level symbols.
To find the final object, the system must:
Symbol3 → Symbol2 → Symbol1 → Object

---

Critical Instructions:

- Use the Rule-based format: Rule 1: ..., Rule 2: ..., etc.
- Each prompt must include exactly 6 mapping rules.
- The order of the rules must be shuffled randomly for each prompt.
- The pairings must vary across examples (e.g., C→A→1 vs. D→B→2).
- Avoid always assigning rules 1–2 to object mappings, rules 3–4 to middle-level, etc. Shuffle freely.

---

Output Format (JSONL):

Each output must be a valid JSON object, with the following fields:
- "ID": A shared identifier for each Question_Aand Question_B pair.
- "Question_A": A prompt describing all 6 rules (random order), ending with: "Please generate the [category] represented by the symbol '[X]'." (e.g., "Please generate the fruit represented by the letter 'C'." or "Please generate the furniture represented by the symbol 'X'.")
- "Question_B": Same as Question_A, but query a different highest-level symbol.
- "Answer": The correct object corresponding to the queried symbol after full resolution.

---

Requirements:

1. Each prompt must define 6 mapping rules, e.g.:
  - 1 represents apple
  - 2 represents orange
  - 'A' maps to 1
  - 'B' maps to 2
  - 'C' maps to 'A'
  - 'D' maps to 'B'
2. The queried symbol must be from the third layer (e.g., 'C', 'D', etc.)
3. The final answer must reflect the three-level resolution:
  - e.g., 'C' → 'A' → 1 → apple
4. The rules must appear in a shuffled order.
5. Use diverse and common objects: apples, spoons, rabbits, notebooks, lamps, etc.
6. Use varied symbols: letters, digits, symbols (A, B, 1, 2, @, #, etc.)
7. Ensure "Question_A" and "Question_B" share exactly the same rules and order.

---

Example Output:

JSON
{"ID": "1", "Question_A": "Rule 1: The number 2 represents oranges. Rule 2: The letter 'A' stands for the number 1. Rule 3: The number 1 represents apples. Rule 4: The letter 'C' refers to the letter 'A'. Rule 5: The letter 'B' stands for the number 2. Rule 6: The letter 'D' refers to the letter 'B'. Please generate the fruit represented by the letter 'C'.", "Answer": "apples"}
{"ID": "1", "Question_B": "Rule 1: The number 2 represents oranges. Rule 2: The letter 'A' stands for the number 1. Rule 3: The number 1 represents apples. Rule 4: The letter 'C' refers to the letter 'A'. Rule 5: The letter 'B' stands for the number 2. Rule 6: The letter 'D' refers to the letter 'B'. Please generate the fruit represented by the letter 'D'.", "Answer": "oranges"}

{"ID": "2", "Question_A": "Rule 1: 9 means chair. Rule 2: 'Y' maps to 9. Rule 3: 'X' maps to 'Y'. Rule 4: 8 means table. Rule 5: 'Z' maps to 8. Rule 6: 'W' maps to 'Z'. Please generate the furniture represented by the symbol 'X'.", "Answer": "chair"}
{"ID": "2", "Question_B": "Rule 1: 9 means chair. Rule 2: 'Y' maps to 9. Rule 3: 'X' maps to 'Y'. Rule 4: 8 means table. Rule 5: 'Z' maps to 8. Rule 6: 'W' maps to 'Z'. Please generate the furniture represented by the symbol 'W'.", "Answer": "table"}

---
Guidelines:
- Avoid repetitive object-symbol mappings across pairs.
- Keep prompts concise but complete and unambiguous.
- Use plain rule-based syntax (Rule x:), but shuffle rule order randomly.
- Ensure all entries follow JSONL format — one JSON object per line, no brackets or explanations.

---
Please generate the data based on these instructions.
Guidelines:
- Avoid repetitive object-symbol mappings across pairs.
- Keep prompts concise but complete and unambiguous.
- Use plain rule-based syntax (Rule x:), but shuffle rule order randomly.
- Ensure all entries follow JSONL format — one JSON object per line, no brackets or explanations.
- Choose familiar objects in daily life, and it's best if there's a significant difference between the two objects.

---
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