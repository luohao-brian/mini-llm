# train_grpo.py 
# TODO: 训练GRPO模型
# 1. 定义一个奖励函数，该函数对完成的文本进行评分。
# 2. 使用训练数据集和奖励函数创建一个GRPOTrainer对象。
# 3. 调用trainer.train()开始训练过程。
# 4. 保存训练好的模型。
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()