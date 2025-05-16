#!/bin/bash
source venv/bin/activate

# 默认前台运行
RUN_IN_BACKGROUND=false
LOG_FILE="training.grpo.log"

# 解析命令行参数
while getopts "dh" opt; do
  case $opt in
    d)
      RUN_IN_BACKGROUND=true
      ;;
    h)
      echo "Usage: $0 [-d] [-h]"
      echo "  -d: Run in background (daemon mode)"
      echo "  -h: Show this help message"
      exit 0
      ;;
    *)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# 环境变量和命令
export OMP_NUM_THREADS=20
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export ACCELERATE_LOG_LEVEL=info

CMD="accelerate launch --config_file recipes/accelerate_configs/grpo_zero.yaml --num_processes 7 recipes/grpo.py --config recipes/grpo.yaml"

# 运行逻辑
if [ "$RUN_IN_BACKGROUND" = true ]; then
  echo "Starting in background mode. Logs will be saved to $LOG_FILE"
  nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
  echo "Process started with PID: $!"
else
  echo "Starting in foreground mode..."
  set -x  # 显示执行的命令（调试模式）
  eval "$CMD"
fi
