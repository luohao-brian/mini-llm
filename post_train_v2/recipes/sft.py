"""监督式微调（SFT）脚本，用于微调解码器语言模型（如GPT、LLaMA等）

使用示例：
# 在8卡H20节点上运行
./sft_start.sh

主要功能：
- 支持多节点分布式训练
- 集成梯度检查点和混合精度训练
- 支持模型权重推送至Hugging Face Hub
- 内置Weights & Biases训练监控
"""

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from recipes.configs import SFTConfig
from recipes.utils import get_model, get_tokenizer
from recipes.utils.callbacks import get_callbacks
from recipes.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    """监督式微调主流程
    
    Args:
        script_args (ScriptArguments): 脚本参数对象，包含数据集路径等配置
        training_args (SFTConfig): 训练参数对象，包含学习率等超参数
        model_args (ModelConfig): 模型参数对象，包含模型架构配置
        
    Raises:
        ValueError: 当检测到无效的检查点或模型配置时抛出
    """
    # 设置随机种子保证实验可复现
    set_seed(training_args.seed)

    ###############
    # 日志系统配置
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],  # 标准输出日志处理器
    )
    # 设置日志级别：DEBUG/INFO/WARNING等
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # 统一设置transformers和datasets库的日志级别
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 打印关键参数信息
    logger.info(f"模型参数 {model_args}")
    logger.info(f"脚本参数 {script_args}")
    logger.info(f"训练参数 {training_args}")

    ########################
    # 检查点恢复机制
    ########################
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"检测到检查点，将从 {last_checkpoint} 恢复训练")

    ####################
    # 训练监控系统初始化
    ####################
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)  # 初始化Weights & Biases集成

    ################
    # 加载数据集
    ################
    dataset = load_dataset(
        script_args.dataset_name, 
        name=script_args.dataset_config  # 数据集配置名称
    )

    ################
    # 初始化分词器
    ################
    tokenizer = get_tokenizer(
        model_args, 
        training_args  # 包含特殊token等配置
    )

    ###################
    # 加载语言模型
    ###################
    logger.info("*** 正在加载模型 ***")
    model = get_model(
        model_args, 
        training_args  # 模型架构参数
    )

    # 设置默认聊天模板
    if tokenizer.chat_template is None:
        logger.info("未提供聊天模板，使用默认ChatML格式")
        model, tokenizer = setup_chat_format(
            model, 
            tokenizer, 
            format="chatml"  # 使用ChatML对话格式
        )

    ############################
    # 初始化SFT训练器
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],  # 训练集划分
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),  # 评估集
        processing_class=tokenizer,  # 数据预处理工具
        peft_config=get_peft_config(model_args),  # PEFT参数高效微调配置
        callbacks=get_callbacks(training_args, model_args),  # 训练回调函数
    )

    ###############
    # 训练主循环
    ###############
    logger.info("*** 开始训练 ***")
    checkpoint = None
    # 检查点恢复逻辑
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    # 执行训练并记录指标
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)  # 记录训练指标
    trainer.save_metrics("train", metrics)  # 保存指标文件
    trainer.save_state()  # 保存训练状态

    ##################################
    # 模型保存与说明文件生成
    ##################################
    logger.info("*** 保存微调后模型 ***")
    trainer.save_model(training_args.output_dir)  # 保存完整模型
    logger.info(f"模型已保存至 {training_args.output_dir}")

    # 在主进程生成模型卡片
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["recipes-r1"],  # 模型仓库标签
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)  # 创建模型卡片
        # 恢复KV缓存配置以加速推理
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    #############
    # 推送模型到Hub
    #############
    if training_args.push_to_hub:
        logger.info("正在推送模型到Hugging Face Hub...")
        trainer.push_to_hub(**kwargs)  # 推送模型和配置文件


if __name__ == "__main__":
    # 参数解析器配置
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    # 解析命令行参数和配置文件
    script_args, training_args, model_args = parser.parse_args_and_config()
    # 执行主流程
    main(script_args, training_args, model_args)