# sft.py
import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description='DPO Training Script')
parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer dataset')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to training dataset')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for sft trained model')
parser.add_argument('--max_length', type=int, required=True, help='input max_length')
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

dataset = load_dataset(args.dataset_path, split="train")
# dataset =dataset.select(range(100)) #截断数据集调试

training_args = SFTConfig(
    max_length=args.max_length,
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    fp16=True,
    save_strategy="epoch",
    logging_steps=100,  # 添加日志步骤以便观察进度
    save_safetensors=True,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()

# model.save_pretrained(save_directory = "/data/CodeRepo/mldemo/output/Qwen2-SFT")
# tokenizer.save_pretrained(save_directory = "/data/CodeRepo/mldemo/output/Qwen2-SFT")
trainer.save_model(output_dir=args.output_dir)
