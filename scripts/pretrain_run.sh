#!/usr/bin/bash

#accelerate launch --multi_gpu  --config_file scripts/accelerate_config/default_config.yaml  pretrain.py --model_path ./models/qwen_0.12B.config  --tokenizer_path ./input/Qwen-tokenizer --dataset_path ./pretrain_data/wiki_pretrain.npy --output_dir ./output/pretain/qwen_0.12B --max_length 1024

accelerate launch --multi_gpu  --config_file scripts/accelerate_config/default_config.yaml  pretrain.py --model_path ./models/qwen2.5_3B.config  --tokenizer_path ./input/Qwen-tokenizer  --output_dir ./output/pretain/qwen2.5_3B.config --max_length 1024
