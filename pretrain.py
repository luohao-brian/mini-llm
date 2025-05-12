import argparse
import multiprocessing
from torch.utils.data import Dataset
import numpy as np
import torch
from loguru import logger
from dataclasses import dataclass
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers import Trainer, TrainingArguments
from transformers import Qwen2Tokenizer

# # 设备设置（保留，确保模型和数据在同一设备）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='DPO Training Script')
parser.add_argument('--model_path', type=str, required=True, help='Path to model config')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer dataset')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to training dataset')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trained model')
parser.add_argument('--max_length', type=int, required=True, help='input max_length')
args = parser.parse_args()

# 直接加载原始配置，**去除所有滑动窗口相关的修改代码**
config = Qwen2Config.from_json_file(args.model_path)
model = Qwen2ForCausalLM(config)
dataset = load_dataset("./input/SkyPile-150B", split="train", streaming=True)
tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_path)

def tokenize_function(example):
    # 添加 padding 参数，将样本填充到最大长度
    tokenized_inputs = tokenizer(example["text"], truncation=True, max_length=4096, padding="max_length")
    # 为因果语言模型将 labels 设置为与 input_ids 相同
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function,
                                batched=True,
                                # num_proc=multiprocessing.cpu_count(),  # 开启流式数据，不能指定 num_proc
                                remove_columns=["text"],
                                )

train_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    fp16=True,
    save_strategy="steps",  # 由于使用 max_steps，将保存策略改为 steps
    save_steps=1000,  # 每 1000 步保存一次模型
    logging_steps=100,  # 添加日志步骤以便观察进度
    save_safetensors=True,
    max_steps=10000,  # 指定最大训练步数，可根据需求调整
    report_to="wandb", #wandb 监控
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(output_dir=args.output_dir)
tokenizer.save_pretrained(save_directory = args.output_dir)

