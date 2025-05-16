"""GRPO训练奖励函数集合

包含多种奖励计算机制：
- 答案准确性验证
- 响应格式规范检查
- 标记计数奖励

各函数设计符合以下原则：
1. 输入输出标准化：统一接收completions参数和关键字参数
2. 异常安全：解析失败返回None避免训练中断
3. 模块化设计：通过注册表机制实现灵活组合
"""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """答案准确性奖励计算函数
    
    Args:
        completions (list[list[dict]]): 模型生成的响应列表，格式为[[{"content": "回答内容"}]]
        solution (list[str]): 标准答案列表，每个元素为字符串格式的正确答案
        
    Returns:
        list[Optional[float]]: 奖励值列表，0/1二值奖励，解析失败返回None
        
    Notes:
        - 使用latex2sympy2进行数学表达式解析
        - 验证过程包含严格的LaTeX格式校验
        - 黄金答案解析失败时会跳过当前样本
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """响应格式规范性奖励函数
    
    Args:
        completions (list): 模型生成的响应列表
        
    Returns:
        list[float]: 奖励值列表，符合格式要求返回1.0，否则0.0
        
    Notes:
        - 使用正则表达式校验响应格式
        - 要求响应包含特定的XML式标记：
           <think>...包裹推理过程
           <answer>...</answer>包裹最终答案
    """
    pattern = r"^<think>\n.*?\n\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """标记计数奖励函数（渐进式奖励）
    
    Args:
        completions (list): 模型生成的响应列表
        
    Returns:
        list[float]: 奖励值列表，取值范围[0.0, 1.0]
        
    Notes:
        - 对每个必需标记给予0.25分奖励：
          1. <think>\n 出现一次
          2. \n\n 出现一次 
          3. \n<answer>\n 出现一次
          4. \n</answer> 出现一次
        - 总分1.0表示所有标记均正确使用
    """
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]

def get_reward_funcs(script_args) -> list[Callable]:
    """动态加载奖励函数
    
    Args:
        script_args: 包含reward_funcs参数的脚本参数对象
        
    Returns:
        list[Callable]: 奖励函数列表
        
    Notes:
        - 依赖REWARD_FUNCS_REGISTRY注册表进行名称映射
        - 当前支持的奖励函数包括：
            tag_count: 标记计数奖励
            accuracy: 准确性奖励
            format: 格式规范性奖励
    """
    REWARD_FUNCS_REGISTRY = {
        "tag_count": tag_count_reward,
        "accuracy": accuracy_reward,
        "format": format_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs