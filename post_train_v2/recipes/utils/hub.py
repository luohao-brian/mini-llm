#!/usr/bin/env python
# coding=utf-8

import logging
import re
from concurrent.futures import Future

from transformers import AutoConfig

from huggingface_hub import (
    create_branch,
    create_repo,
    get_safetensors_metadata,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    repo_exists,
    upload_folder,
)
from trl import GRPOConfig, SFTConfig


logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args: SFTConfig | GRPOConfig, extra_ignore_patterns=[]) -> Future:
    """推送模型检查点到Hugging Face Hub指定分支
    
    Args:
        training_args: 训练配置参数（SFTConfig或GRPOConfig类型）
        extra_ignore_patterns: 需要忽略的额外文件模式列表
        
    Returns:
        Future: 异步上传任务的Future对象
        
    Steps:
        1. 创建/验证Hugging Face仓库
        2. 基于初始提交创建新分支
        3. 上传指定目录内容到分支
    """
    # 创建仓库（如果不存在）
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    
    # 获取初始提交作为分支基准
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    
    # 创建目标分支
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    
    # 设置文件忽略模式（不推送检查点和优化器状态）
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision}...")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    
    # 执行异步上传
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} successfully!")

    return future


def check_hub_revision_exists(training_args: SFTConfig | GRPOConfig):
    """检查指定修订版本是否已存在
    
    Args:
        training_args: 包含仓库配置的训练参数
        
    Raises:
        ValueError: 当修订已存在且未启用覆盖选项时抛出
    """
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # 获取现有分支列表
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]
            
            # 检查目标修订是否存在
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id,
                    revision=training_args.hub_model_revision,
                )
                
                # 如果存在README且未启用覆盖，抛出异常
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: str) -> int:
    """从仓库ID解析模型参数数量
    
    Args:
        repo_id: 模型仓库标识符（支持格式如"Qwen-1.8b"、"8x7b"等）
        
    Returns:
        int: 参数总数（单位：百万/十亿），无法解析时返回-1
        
    Implementation:
        1. 优先读取safetensors元数据
        2. 若失败则使用正则匹配模型规模模式
    """
    try:
        # 方法1：通过safetensors元数据获取
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except Exception:
        # 方法2：正则匹配模型规模模式
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for full_match, number1, _, _, number2, _, unit in matches:
            # 处理乘积格式（如8x7b）
            if number2:
                number = float(number1) * float(number2)
            else:
                number = float(number1)

            # 单位转换
            multiplier = 1_000_000_000 if unit == "b" else 1_000_000
            param_counts.append(int(number * multiplier))

        return max(param_counts) if param_counts else -1


def get_gpu_count_for_vllm(model_name: str, revision: str = "main", num_gpus: int = 8) -> int:
    """计算适配vLLM框架的最佳GPU数量
    
    vLLM要求注意力头数能被GPU数整除，且64能被GPU数整除
    
    Args:
        model_name: 模型名称
        revision: 模型版本（默认main）
        num_gpus: 初始GPU数量
        
    Returns:
        int: 调整后的有效GPU数量
    """
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    
    # 获取注意力头数
    num_heads = config.num_attention_heads
    
    # 动态调整GPU数量
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(f"Reducing num_gpus from {num_gpus} to {num_gpus - 1} to make num_heads divisible by num_gpus")
        num_gpus -= 1
        
    return num_gpus