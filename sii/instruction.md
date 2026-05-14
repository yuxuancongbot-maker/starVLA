# FlowerVLA 环境配置指南

## 前置条件

- 操作系统：Linux x86_64
- GPU：NVIDIA RTX 4090 / H200（CUDA 12.8 驱动已装）
- conda 已安装于 `$HOME/miniforge3`（Miniforge3）

## 第一步：创建 starVLA conda 环境

```bash
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -n starVLA python=3.10 -y
conda activate starVLA
```

## 第二步：安装依赖（绕过全局 pip 约束）

> **重要**：本机 `/etc/pip/constraint.txt` 锁死了 NVIDIA 定制版 PyTorch 版本，与 starVLA 要求的版本冲突。
> 必须设置 `PIP_CONSTRAINT=` 绕过，且使用 `python -m pip`（而非裸 `pip`）确保包安装到 conda 环境。

### 2.1 先装 PyTorch（CUDA 12.4）

```bash
PIP_CONSTRAINT= python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 2.2 装其余依赖

```bash
PIP_CONSTRAINT= python -m pip install \
  transformers==4.57.0 \
  accelerate==1.5.2 \
  tiktoken \
  transformers_stream_generator==0.0.4 \
  setuptools==80.9.0 \
  websocket \
  websocket-client==1.8.0 \
  albumentations==1.4.18 \
  pydantic==2.10.6 \
  pyarrow==14.0.1 \
  fastparquet==2024.11.0 \
  av==12.3.0 \
  numpydantic==1.6.9 \
  deepspeed==0.16.9 \
  qwen-vl-utils \
  omegaconf \
  wandb \
  rich \
  diffusers \
  timm \
  tyro \
  websockets \
  tdigest==0.5.2.2 \
  scipy \
  einops
```

### 2.3 安装 flash-attn（可选，加速训练）

```bash
PIP_CONSTRAINT= python -m pip install flash-attn --no-build-isolation
```

> 如编译失败可跳过，不影响 FlowerVLA 推理和微调。

### 2.4 以 editable 模式安装 starVLA

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/starvla
PIP_CONSTRAINT= python -m pip install -e .
```

## 第三步：验证环境

```bash
# 确认关键包版本
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import transformers; print(f'transformers {transformers.__version__}')
import timm; print(f'timm {timm.__version__}')
import omegaconf; print('omegaconf OK')
import deepspeed; print('deepspeed OK')
import accelerate; print('accelerate OK')
print('All checks passed!')
"
```

## 第四步：验证 FlowerVLA 框架可导入

```bash
# 测试框架类是否能被自动发现
python -c "
from starVLA.model.framework.base_framework import _auto_import_framework_modules, FRAMEWORK_REGISTRY
_auto_import_framework_modules()
print('Available frameworks:', sorted(FRAMEWORK_REGISTRY._registry.keys()))
assert 'FlowerVLA' in FRAMEWORK_REGISTRY._registry, 'FlowerVLA not registered!'
print('FlowerVLA registered successfully!')
"
```

## 常见问题

### Q: pip 安装时出现 torch/torchvision 版本冲突？

A: 确保每次 pip install 前面加了 `PIP_CONSTRAINT=`，这会绕过 `/etc/pip/constraint.txt` 的全局版本锁定。

### Q: deepspeed 安装很慢？

A: deepspeed 需要编译 CUDA kernel，可能耗时 5-10 分钟。如果不需要 deepspeed（单卡推理），可以暂时跳过。

### Q: 找不到 `pipablepytorch3d==0.7.6`？

A: 该包仅用于 3D 渲染评测，FlowerVLA 不依赖它，可安全跳过。

### Q: 如何激活已有环境？

```bash
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate starVLA
```
