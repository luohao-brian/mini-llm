import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """获取并配置预训练模型的分词器
    
    Args:
        model_args: 模型配置参数，包含模型路径等信息
        training_args: 训练配置参数，包含聊天模板配置
        
    Returns:
        PreTrainedTokenizer: 初始化完成的分词器实例
        
    Features:
        - 支持远程代码加载 (trust_remote_code)
        - 可自定义聊天模板 (chat_template)
    """
    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 应用自定义聊天模板（如果配置）
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """加载并配置预训练语言模型
    
    Args:
        model_args: 模型配置参数（量化、注意力机制等）
        training_args: 训练配置参数（梯度检查点等）
        
    Returns:
        AutoModelForCausalLM: 初始化完成的因果语言模型
        
    Features:
        - 自动数据类型转换 (torch_dtype)
        - 支持4/8-bit量化
        - 注意力机制优化选择
        - 梯度检查点兼容性处理
    """
    # 解析数据类型配置（自动或指定类型）
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] 
        else getattr(torch, model_args.torch_dtype)
    )
    
    # 获取量化配置（None表示不启用量化）
    quantization_config = get_quantization_config(model_args)
    
    # 构建模型加载参数
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,  # 允许加载自定义模型代码
        attn_implementation=model_args.attn_implementation,  # 注意力机制实现方式（如flash_attention2）
        torch_dtype=torch_dtype,  # 模型数据类型（自动/float16/bfloat32等）
        use_cache=False if training_args.gradient_checkpointing else True,  # 梯度检查点与KV缓存兼容性
        device_map=get_kbit_device_map() if quantization_config is not None else None,  # 量化设备映射
        quantization_config=quantization_config,  # 4/8-bit量化配置
    )
    
    # 从预训练路径加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model