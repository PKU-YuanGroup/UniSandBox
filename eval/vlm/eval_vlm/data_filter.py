#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import os
import random
import re

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm


FILTER_INSTRUCTION = (
    "You are an expert image quality assessor. Your task is to evaluate an image based on a specific question. You must be extremely strict and analytical in your evaluation.\n\n"
    "Question: {question}\n\n"
    "Please follow these steps exactly:\n\n"
    "1.  **Analyze the Question**:\n"
    "    * First, identify the **type of object(s)** mentioned in the question (e.g., desks, chairs, cats, etc.).\n"
    "    * Next, determine the **correct number** of objects required by the question. If the question contains a mathematical expression (e.g., '8 / 4', '5 - 3'), you MUST perform the calculation first to get the correct number.\n\n"
    "2.  **Examine the Image and Count Objects**:\n"
    "    * Carefully examine the image and count the number of objects of the identified type.\n"
    "    * Describe the objects you see and state the actual count.\n\n"
    "3.  **Perform a Strict Criteria Check**:\n"
    "    * Based on your analysis and count, check if the image meets **ALL** of the following criteria. **Be very strict.**\n"
    "        * **Correct Number**: The actual count of objects in the image **MUST match** the correct number you determined in step 1.\n"
    "        * **Correct Type**: All objects found must be of the correct type as specified in the question.\n"
    "        * **No Extra Objects**: The image should not contain any other objects that are not mentioned in the question.\n"
    "        * **Clear Quality**: The image must be clear, recognizable, and free from blurriness or distortion.\n"
    "        * **Complete Objects**: The objects must be complete and uncut, with no parts missing or obscured.\n\n"
    "4.  **Final Judgment**:\n"
    "    * After completing the checks in step 3, provide your final judgment using **ONLY** one of the two formats below. Do not add any extra text or explanation.\n"
    "    * If all criteria are met: `Final Answer: YES`\n"
    "    * If even ONE criterion is NOT met: `Final Answer: NO`\n\n"
    "Think step by step and be very strict in your evaluation."
)


def extract_judgment(text):
    match = re.search(r'Final Answer:\s*(YES|NO)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    if re.search(r'\bYES\b', text, re.IGNORECASE) and not re.search(r'\bNO\b', text, re.IGNORECASE):
        return "YES"
    elif re.search(r'\bNO\b', text, re.IGNORECASE) and not re.search(r'\bYES\b', text, re.IGNORECASE):
        return "NO"
    else:
        return text.strip()[:50]


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversations = [_['conversations'] for _ in batches]
    item_ids = [_['item_id'] for _ in batches]
    original_answers = [_['original_answer'] for _ in batches]
    image_paths = [_['image_path'] for _ in batches]

    return questions, images, conversations, item_ids, original_answers, image_paths


class FilterDataset(torch.utils.data.Dataset):

    def __init__(self, data_file):
        self.data_file = data_file
        self.data = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        question = data['Question']
        original_answer = data['Answer']
        item_id = data['id']

        image_path = data['image']
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path).convert('RGB')
        images = [image]

        filter_prompt = FILTER_INSTRUCTION.format(question=question)
        
        images, conversation = process_conversation(images, filter_prompt)

        return {
            'item_id': item_id,
            'question': question,
            'original_answer': original_answer,
            'images': images,
            'conversations': conversation,
            'image_path': image_path,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def filter_data():
    random.seed(args.seed)

    dataset = FilterDataset(
        data_file=args.data_file,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    accepted_items = []
    rejected_items = []
    
    for _, (questions, images, conversations, item_ids, original_answers, image_paths) in tqdm(enumerate(dataloader)):
        pred = model.chat(
            tokenizer, 
            new_token_ids,
            image_transform,
            images=images[0], # batch=1
            prompt=conversations[0], # batch=1
            max_length=args.max_new_tokens,
        )
        
        judgment = extract_judgment(pred)
        
        for item_id, question, original_answer, image_path in zip(item_ids, questions, original_answers, image_paths):
            result = {
                'id': item_id,
                'Question': question,
                'Answer': original_answer,
                'image': image_path,
                'model_judgment': judgment,
                'model_response': pred,
                'accepted': judgment == "YES"
            }
            
            outputs.append(result)
            
            if judgment == "YES":
                accepted_items.append({
                    'Question': question,
                    'Answer': original_answer,
                    'id': item_id,
                    'image': image_path
                })
            else:
                rejected_items.append({
                    'Question': question,
                    'Answer': original_answer,
                    'id': item_id,
                    'image': image_path,
                    'reason': pred
                })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    merged_accepted = [None for _ in range(world_size)]
    merged_rejected = [None for _ in range(world_size)]
    
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))
    torch.distributed.all_gather_object(merged_accepted, json.dumps(accepted_items))
    torch.distributed.all_gather_object(merged_rejected, json.dumps(rejected_items))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    
    merged_accepted = [json.loads(_) for _ in merged_accepted]
    merged_accepted = [_ for _ in itertools.chain.from_iterable(merged_accepted)]
    
    merged_rejected = [json.loads(_) for _ in merged_rejected]
    merged_rejected = [_ for _ in itertools.chain.from_iterable(merged_rejected)]

    if torch.distributed.get_rank() == 0:
        print(f'Filtering completed!')
        print(f'Total samples: {len(merged_outputs)}')
        print(f'Accepted samples: {len(merged_accepted)}')
        print(f'Rejected samples: {len(merged_rejected)}')
        print(f'Acceptance rate: {len(merged_accepted)/len(merged_outputs)*100:.2f}%')
        
        results_file = os.path.join(args.out_dir, 'filter_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(merged_outputs, f, ensure_ascii=False, indent=2)
        print(f'Detailed results saved to: {results_file}')
        
        accepted_file = os.path.join(args.out_dir, 'filtered_data.jsonl')
        with open(accepted_file, 'w', encoding='utf-8') as f:
            for item in merged_accepted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Filtered data saved to: {accepted_file}')
        
        rejected_file = os.path.join(args.out_dir, 'rejected_data.jsonl')
        with open(rejected_file, 'w', encoding='utf-8') as f:
            for item in merged_rejected:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Rejected data saved to: {rejected_file}')
        
        stats = {
            'total_samples': len(merged_outputs),
            'accepted_samples': len(merged_accepted),
            'rejected_samples': len(merged_rejected),
            'acceptance_rate': len(merged_accepted)/len(merged_outputs)*100,
            'data_file': args.data_file,
            'model_path': args.model_path,
        }
        stats_file = os.path.join(args.out_dir, 'filter_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f'Statistics saved to: {stats_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True, 
                        help='Input jsonl data file path containing image field')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='filter_results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--max-new-tokens', type=int, default=800)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[filter] total_params: {total_params}B')

    filter_data() 