"""GRPO（基于梯度惩罚的奖励优化）模型训练主流程

该脚本实现了一个完整的GRPO模型训练流程，包含以下核心功能：
- 实验可重复性配置
- 分布式训练日志管理
- 模型与分词器加载
- 奖励函数集成
- 对话数据格式化
- 训练循环与评估
- 模型检查点管理
- Hugging Face Hub集成

典型使用场景：
1. 使用自定义数据集进行强化学习对齐训练
2. 多GPU/TPU分布式训练
3. 结合PEFT(参数高效微调)方法进行高效训练

依赖项：
- Hugging Face Transformers >=4.40.0
- TRL (Transformer Reinforcement Learning) 库
- PEFT (Parameter-Efficient Fine-Tuning) 库
- WandB (可选，用于实验追踪)

示例用法：
   第一步启动推理服务：
   ./vllm_serve_start.sh
   第二步启动训练任务:
  ./grpo_start.sh

"""

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from recipes.configs import GRPOConfig, GRPOScriptArguments
from recipes.rewards import get_reward_funcs
from recipes.utils import get_model, get_tokenizer
from recipes.utils.callbacks import get_callbacks
from recipes.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args):
    """GRPO训练主流程
    
    Args:
        script_args (GRPOScriptArguments): 脚本特有参数配置对象
            - dataset_name: 数据集名称或路径
            - dataset_config: 数据集配置名称
            - dataset_prompt_column: 输入文本字段名
            - dataset_train_split: 训练集划分名称
            - dataset_test_split: 验证集划分名称
            
        training_args (GRPOConfig): 训练参数配置对象
            - seed: 随机种子
            - output_dir: 输出目录
            - fp16: 是否使用16位精度
            - resume_from_checkpoint: 检查点恢复路径
            - eval_strategy: 评估策略
            - push_to_hub: 是否推送模型到HF Hub
            
        model_args (ModelConfig): 模型参数配置对象
            - model_name_or_path: 模型标识或本地路径
            - peft_type: PEFT类型 (LORA, P_TUNING等)
            - torch_dtype: 张量精度类型

    Raises:
        ValueError: 当数据集中缺少指定的prompt字段时
    """
    
    # 设置随机种子保证实验可重复性
    set_seed(training_args.seed)

    # 日志系统配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # 设置三方库日志级别
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 记录硬件环境信息
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    # 打印配置参数摘要
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # 检查点恢复机制
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # 初始化WandB实验追踪
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # 加载数据集
    """支持的格式：
    - HuggingFace Hub数据集
    - 本地JSON/CSV文件
    - 内存中的字典格式
    """
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # 初始化分词器
    """分词器特性：
    - 自动处理特殊token
    - 支持左右填充配置
    - 兼容模型的最大长度限制
    """
    tokenizer = get_tokenizer(model_args, training_args)

    # 加载预训练模型
    """模型加载选项：
    1. 从HuggingFace Hub加载
    2. 从本地缓存加载
    3. 使用PEFT配置加载适配器
    """
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # 获取奖励函数集合
    """支持的奖励类型：
    - 人工标注奖励
    - 基于规则的奖励
    - 模型预测奖励
    """
    reward_funcs = get_reward_funcs(script_args)

    # 格式化对话数据
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        """将原始数据转换为对话格式
        
        Args:
            example (dict): 单条数据样本
            prompt_column (str): 输入文本字段名
            
        Returns:
            dict: 包含对话角色的字典
            
        Example 输出格式:
        {
            "prompt": [
                {"role": "system", "content": "你是有帮助的助手"},
                {"role": "user", "content": "如何做蛋糕？"}
            ]
        }
        """
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # 清理冗余字段
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 初始化GRPO训练器
    """关键组件：
    - reward_funcs: 奖励计算模块
    - peft_config: 参数高效微调配置
    - callbacks: 训练回调函数（如早停、日志等）
    """
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # 训练阶段
    logger.info("***  开始训练 ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    # 执行训练循环
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 记录训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 模型保存
    """保存内容包含：
    - 完整模型权重（或PEFT适配器）
    - 分词器配置
    - 训练参数配置
    - 模型卡片信息
    """
    logger.info("*** 保存模型 ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # 在主进程保存附加信息
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["recipes-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # 恢复推理优化配置
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # 模型推送
    if training_args.push_to_hub:
        logger.info("正在推送模型到Hugging Face Hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # 参数解析系统
    """支持的配置类型：
    1. GRPOScriptArguments: 脚本运行参数
    2. GRPOConfig: 训练策略参数  
    3. ModelConfig: 模型架构参数
    """
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)