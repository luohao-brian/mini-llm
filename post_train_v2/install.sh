#!/bin/bash
set -exo pipefail

# ====================== 环境检查 ======================
# 1. 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "错误：未安装 uv 工具链，请先执行: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 2. 严格检查 CUDA 12.4
if ! nvcc --version | grep -q "release 12.4"; then
    echo "错误：需 CUDA 12.4，当前版本：$(nvcc --version | grep release)"
    exit 1
fi

# ====================== 参数配置 ======================
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}  # 硬编码指定12.4
export PATH="$CUDA_HOME/bin:$PATH"

# ====================== 安装流程 ======================
# 1. 创建虚拟环境
uv venv venv --python 3.11 || { echo "虚拟环境创建失败"; exit 1; }
source venv/bin/activate
uv pip install --upgrade pip

# 2. 强制安装 setuptools（覆盖系统旧版）
uv pip install --upgrade "setuptools>=68.0.0" wheel || { 
    echo "setuptools 安装失败"; 
    exit 1
}

# 3. 安装 requirements.txt 中的依赖
uv pip install -r requirements.txt || { 
    echo "依赖安装失败";
    exit 1
}

# 4. 单独安装 flash-attn（禁用构建隔离）
uv pip install flash-attn \
    --no-build-isolation \
    || { echo "flash-attn 编译失败"; exit 1; }

# ====================== 验证安装 ======================
python -c "
import torch, flash_attn
assert torch.cuda.is_available(), 'Torch CUDA 不可用'
assert torch.version.cuda.startswith('12.4'), f'CUDA 版本不符: {torch.version.cuda}'
print(f'\n\033[32m成功安装: torch={torch.__version__}, flash_attn={flash_attn.__version__}\033[0m')
"
