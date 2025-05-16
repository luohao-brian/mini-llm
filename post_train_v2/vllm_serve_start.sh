#!/bin/bash
source venv/bin/activate

# 默认配置
LOG_FILE="vllm_serve.log"
MODEL_NAME="./DeepSeek-R1-Distill-Qwen-1.5B"
GPU_DEVICE="0"

# 前台运行服务
run_foreground() {
    echo "[INFO] 启动 vLLM 服务（前台模式）"
    echo "[INFO] 使用模型: $MODEL_NAME"
    echo "[INFO] 使用GPU设备: $GPU_DEVICE"
    
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE trl vllm-serve --model $MODEL_NAME
}

# 后台运行服务
run_background() {
    echo "[INFO] 启动 vLLM 服务（后台模式）"
    echo "[INFO] 使用模型: $MODEL_NAME"
    echo "[INFO] 使用GPU设备: $GPU_DEVICE"
    echo "[INFO] 日志输出到: $LOG_FILE"
    
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_DEVICE trl vllm-serve --model $MODEL_NAME" > $LOG_FILE 2>&1 &
    echo "[INFO] 服务已启动，PID: $!"
}

# 显示帮助信息
show_help() {
    echo "vLLM 服务启动脚本"
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -d, --daemon   后台模式运行"
    echo "  -g, --gpu     指定GPU设备（默认: 0）"
    echo "  -m, --model   指定模型名称"
    echo "  -h, --help    显示帮助信息"
}

# 参数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--daemon)
            MODE="background"
            shift
            ;;
        -g|--gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知选项 $1"
            show_help
            exit 1
            ;;
    esac
done

# 主执行逻辑
case "$MODE" in
    "background")
        run_background
        ;;
    *)
        run_foreground
        ;;
esac
