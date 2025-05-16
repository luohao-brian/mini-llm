#!/bin/bash
source venv/bin/activate

set -x

# 用法示例：./eval.sh --model Qwen/Qwen2.5-3B-Instruct --task aime24 --gpus 8

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --gpus) NUM_GPUS="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 参数校验
if [[ -z "$MODEL" || -z "$TASK" || -z "$NUM_GPUS" ]]; then
    echo "用法: $0 --model <模型路径> --task <任务名> --gpus <GPU数量>"
    echo "示例: $0 --model Qwen/Qwen2.5-3B-Instruct --task math_500 --gpus 8"
    exit 1
fi

# 构造模型参数
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,\
max_model_length=32768,gpu_memory_utilization=0.8,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# 创建输出目录
OUTPUT_DIR="data/evals/${MODEL//\//_}"  # 替换/为_避免路径问题
mkdir -p "$OUTPUT_DIR"

# 执行评估
lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"
