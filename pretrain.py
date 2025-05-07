from torch.utils.data import Dataset
import numpy as np
import torch
from loguru import logger
from dataclasses import dataclass
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers import Trainer, TrainingArguments

# 设备设置（保留，确保模型和数据在同一设备）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Args:
    max_length: int

class MyDataset(Dataset):
    """加载数据"""
    def __init__(self, filenames, args):
        self.data = []
        self.index_map = {}
        self.token_size, self.smp_size = 0, 0
        data_lst = []
        for _, filename in enumerate(filenames):
            data = np.load(filename)
            data_lst.append(data)
        data = np.concatenate(data_lst)
        data = data[: args.max_length * int(len(data) / args.max_length)]
        self.data = data.reshape(-1, args.max_length)

        self.token_size = self.data.shape[0] * args.max_length
        self.sample_size = self.data.shape[0]
        logger.info(f"token_size: {self.token_size}, smp_size: {self.sample_size}")

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return {
            "input_ids": torch.from_numpy(X),
            "labels": torch.from_numpy(Y),
        }

args = Args(max_length=1024)

# 直接加载原始配置，**去除所有滑动窗口相关的修改代码**
config = Qwen2Config.from_json_file("./qwen_0.12B.config")
model = Qwen2ForCausalLM(config).to(device)  # 模型转移到设备
print(model)

train_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    fp16=True,
    save_strategy="epoch",
    logging_steps=100,  # 添加日志步骤以便观察进度
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=MyDataset(["./pretrain_data/wiki_pretrain.npy"], args=args),
)

trainer.train()

#dataset = MyDataset(["./pretrain_data/wiki_pretrain.npy"], args=args)
#for i in range(10):  # 取前10个样本进行测试
#    sample = dataset[i]
#    input_ids = sample["input_ids"].unsqueeze(0).to(device)
#    labels = sample["labels"].unsqueeze(0).to(device)
#    try:
#        outputs = model(input_ids=input_ids, labels=labels)
#        print(f"Loss: {outputs.loss.item()}")
#    except Exception as e:
#        print(f"Error occurred during forward pass: {e}")

