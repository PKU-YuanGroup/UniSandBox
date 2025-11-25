# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    # 'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {

        "math1_reject_5k":{
            'data_dir': 'path/to/math1_reject_5k',
            'num_files': 1,
            'num_total_samples': 5000,
        },
        "math2_reject_5k": {
            'data_dir': 'path/to/math2_reject_5k',
            'num_files': 1,
            'num_total_samples': 5000,
        },
        "math3_reject_5k": {
            'data_dir': 'path/to/math3_reject_5k',
            'num_files': 1,
            'num_total_samples': 5000,
        },
        "mapping1_1w_reject": {
            'data_dir': 'path/to/mapping1_1w_reject',
            'num_files': 1,
            'num_total_samples': 10000,
        },
        "mapping2_1w_reject": {
            'data_dir': 'path/to/mapping2_1w_reject',
            'num_files': 1,
            'num_total_samples': 10000,
        },
        "mapping3_1w_reject": {
            'data_dir': 'path/to/mapping3_1w_reject',
            'num_files': 1,
            'num_total_samples': 10000,
        },
      
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
        'Lysendria': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Lysendria.jsonl',
			'num_total_samples': 40
        },
        'Kaelorix': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Kaelorix.jsonl',
			'num_total_samples': 40
        },
        'Jovianne': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Jovianne.jsonl',
			'num_total_samples': 40
        },
        'Zefyria': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Zefyria.jsonl',
			'num_total_samples': 40
        },
        'Aurelius': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Aurelius.jsonl',
			'num_total_samples': 40
        },
        'Nyxella': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Nyxella.jsonl',
			'num_total_samples': 40
        },
        'Valerian': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Valerian.jsonl',
			'num_total_samples': 40
        },
        'Thalassia': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Thalassia.jsonl',
			'num_total_samples': 40
        },
        'Orionax': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Orionax.jsonl',
			'num_total_samples': 40
        },
        'Evandriel': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Evandriel.jsonl',
			'num_total_samples': 40
        },
        'Lysendria_Kaelorix': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Lysendria_Kaelorix.jsonl',
			'num_total_samples': 80
        },
        'Jovianne_Zefyria': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Jovianne_Zefyria.jsonl',
			'num_total_samples': 80
        },
        'Aurelius_Nyxella': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Aurelius_Nyxella.jsonl',
			'num_total_samples': 80
        },
        'Valerian_Thalassia': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Valerian_Thalassia.jsonl',
			'num_total_samples': 80
        },
        'Orionax_Evandriel': {
            'data_dir': 'path/to/images',
			'jsonl_path': './data/knowledge/Orionax_Evandriel.jsonl',
			'num_total_samples': 80
        },
        
    },
}