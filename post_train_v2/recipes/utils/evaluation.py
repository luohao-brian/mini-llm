import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id

if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig  # 类型检查专用导入

import base64
import os

# vLLM在Slurm训练作业中运行的特殊环境配置
# 参考代码: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# Slack讨论: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")  # 获取用户主目录路径
VLLM_SLURM_PREFIX = [  # Slurm作业提交前缀命令
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch ",
]

def register_lighteval_task(
    configs: Dict[str, str],
    eval_suite: str,
    task_name: str,
    task_list: str,
    num_fewshot: int = 0,
):
    """注册LightEval评测任务配置
    
    - 核心任务参考: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    - 自定义任务需存储在 scripts/evaluation/extended_lighteval_tasks
    
    Args:
        configs (Dict[str, str]): 存储任务配置的字典
        eval_suite (str): 评测套件名称(lighteval/extended)
        task_name (str): 任务配置名称
        task_list (str): 逗号分隔的任务列表，格式为"套件类型|任务名|fewshot数|版本"
        num_fewshot (int): few-shot示例数量，默认为0
    """
    # 将任务列表转换为lighteval格式
    formatted_tasks = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = formatted_tasks

# 初始化LightEval任务字典
LIGHTEVAL_TASKS = {}

# 注册预定义评测任务
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "lighteval", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)

def get_lighteval_tasks():
    """获取所有已注册的LightEval任务名称列表
    
    Returns:
        list: 包含所有任务名称的列表
    """
    return list(LIGHTEVAL_TASKS.keys())

# 支持的基准测试列表
SUPPORTED_BENCHMARKS = get_lighteval_tasks()

def run_lighteval_job(
    benchmark: str,
    training_args: Union["SFTConfig", "GRPOConfig"],
    model_args: "ModelConfig",
) -> None:
    """启动LightEval评测任务的Slurm作业
    
    Args:
        benchmark (str): 基准测试名称
        training_args: 训练配置参数
        model_args: 模型配置参数
        
    Raises:
        subprocess.CalledProcessError: 当Slurm作业提交失败时抛出
    """
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    
    # 动态调整GPU资源配置
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    
    # 大模型(>=30B参数)或数学类评测使用张量并行
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 8  # 默认使用8卡
        tensor_parallel = False

    # 构建Slurm命令
    cmd = VLLM_SLURM_PREFIX.copy()
    cmd_args = [
        f"--gres=gpu:{num_gpus}",  # GPU资源申请
        f"--job-name=or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}",  # 作业名称
        "slurm/evaluate.slurm",  # Slurm脚本路径
        benchmark,  # 评测基准名称
        f'"{task_list}"',  # 任务列表(带引号防止空格问题)
        model_name,  # 模型仓库ID
        model_revision,  # 模型版本
        f"{tensor_parallel}",  # 是否启用张量并行
        f"{model_args.trust_remote_code}",  # 是否信任远程代码
    ]
    
    # 处理系统提示词(base64编码避免特殊字符问题)
    if training_args.system_prompt is not None:
        prompt_encoded = base64.b64encode(training_args.system_prompt.encode()).decode()
        cmd_args.append(prompt_encoded)
    
    # 组合完整命令
    cmd[-1] += " " + " ".join(cmd_args)
    
    # 提交Slurm作业
    subprocess.run(cmd, check=True)

def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    """执行所有指定的基准测试
    
    Args:
        training_args: 包含评测列表的训练配置
        model_args: 模型配置参数
        
    Raises:
        ValueError: 当传入不支持的基准测试名称时抛出
    """
    benchmarks = training_args.benchmarks
    
    # 处理'all'特殊参数
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()  # 获取所有支持的任务
    
    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")