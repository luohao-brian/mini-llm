#!/usr/bin/env python
# coding=utf-8

import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    """检查系统是否支持SLURM队列管理系统
    
    Returns:
        bool: 如果存在sinfo命令返回True，否则返回False
    """
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    """虚拟配置容器类
    
    用于动态创建配置对象，避免修改原始训练配置
    
    Args:
        **kwargs: 接受任意关键字参数作为配置属性
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    """模型检查点推送回调
    
    在训练保存检查点时，将模型推送到Hugging Face Hub，
    并可选触发基准测试任务
    
    Args:
        model_config: 模型配置对象
    """
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """保存检查点时的回调处理
        
        Args:
            args: 训练参数对象
            state: 训练状态对象
            control: 训练控制对象
        """
        if state.is_world_process_zero:
            global_step = state.global_step

            # 使用虚拟配置避免破坏加速器的分布式状态
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            # 异步推送检查点到Hub
            future = push_to_hub_revision(
                dummy_config, 
                extra_ignore_patterns=["*.pt"]  # 排除优化器状态文件
            )

            # 如果支持SLURM，添加基准测试回调
            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    """推送完成后的回调函数"""
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)


# 可用回调函数注册表
CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    """根据配置获取回调实例列表
    
    Args:
        train_config: 训练配置对象
        model_config: 模型配置对象
        
    Returns:
        List[TrainerCallback]: 回调实例列表
        
    Raises:
        ValueError: 当配置中指定了未注册的回调时抛出
    """
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks