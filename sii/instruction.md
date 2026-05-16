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

---

## 第五步：FlowerVLA 在 CALVIN 上的推理评测

CALVIN 评测需要**两个终端**同时运行：一个启动模型推理服务器（starVLA 环境），另一个运行 CALVIN 环境客户端（calvin 环境）。

### 前置准备

- starVLA conda 环境已配置（见第一步～第四步）
- calvin conda 环境已配置（需含 `calvin_env`、`calvin_agent`、`hydra`、`omegaconf`、`moviepy`）
  - **重要**：calvin_venv 必须使用 `numpy<2`（torch 1.13.1 不兼容 numpy 2.x）
  - **重要**：calvin_venv 必须安装 `moviepy==1.0.3`（2.x 移除了 `moviepy.editor`）
- **系统依赖**：无头服务器需安装 OSMesa：`apt-get install -y libosmesa6-dev`
- FLOWER checkpoint 权重文件：`checkpoints/flower_calvin_abcd/model.safetensors`
- starVLA 格式 checkpoint 目录已创建：`checkpoints/flower_calvin_abcd_starvla/`
- CALVIN 数据集路径（如 `/path/to/calvin/task_D_D/`）

### 5.1 确认 starVLA checkpoint 目录结构

```
checkpoints/flower_calvin_abcd_starvla/
├── config.yaml                          # starVLA 格式，framework.name: FlowerVLA
├── dataset_statistics.json              # identity stats（min=-1, max=1, robot_type: libero_franka）
└── checkpoints/
    └── model.safetensors -> ../../flower_calvin_abcd/model.safetensors
```

### 5.2 终端 1：启动 FlowerVLA 推理服务器

```bash
# 激活 starVLA 环境
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate starVLA

# 进入项目根目录
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/starvla

# 设置 PYTHONPATH（**必须**，否则报 ModuleNotFoundError: No module named 'deployment'）
export PYTHONPATH=$(pwd):${PYTHONPATH}

# 启动推理服务器
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
    --ckpt_path checkpoints/flower_calvin_abcd_starvla/checkpoints/model.safetensors \
    --port 5694
```

**参数说明**：
| 参数 | 值 | 说明 |
|---|---|---|
| `--ckpt_path` | `checkpoints/flower_calvin_abcd_starvla/checkpoints/model.safetensors` | 指向 safetensors 文件（`read_mode_config` 会自动向上找 `config.yaml` + `dataset_statistics.json`） |
| `--port` | `5694` | WebSocket 服务端口 |

> **注意**：不要使用 `--use_bf16`。FlowerVLA 的 Florence-2 VLM + DiT head 在 bf16 下存在 dtype 不兼容（`expected scalar type Float but found BFloat16`），需用 float32 推理。Florence-2-large 约 0.78B，float32 单卡显存足够。

**预期输出**：服务器启动后打印 `PolicyServerWrapper` 加载日志，包括 key remapping 信息，最后显示 `WebsocketPolicyServer started on 0.0.0.0:5694`。

### 5.3 终端 2：运行 CALVIN 评测客户端

```bash
# 激活 calvin 环境（请根据实际路径调整）
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate calvin_venv

# 进入项目根目录
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/starvla

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd):${PYTHONPATH}

# 以下路径已指向本机实际文件
CALVIN_DATASET_PATH=/inspire/hdd/global_user/lijinhao-240108540148/research/dual_sys_VLA/calvin/dataset/task_ABC_D/
CALVIN_CONFIG_PATH=/inspire/hdd/global_user/lijinhao-240108540148/research/dual_sys_VLA/calvin/calvin_models/conf/
EVAL_SEQUENCES_PATH=examples/calvin/eval_files/eval_sequences.json

# 创建日志目录
LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

# 运行评测
python ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path checkpoints/flower_calvin_abcd_starvla/checkpoints/model.safetensors \
    --args.unnorm-key libero_franka \
    --args.host 127.0.0.1 \
    --args.port 5694 \
    --args.dataset_path ${CALVIN_DATASET_PATH} \
    --args.calvin_config_path ${CALVIN_CONFIG_PATH} \
    --args.eval_sequences_path ${EVAL_SEQUENCES_PATH} \
    --args.num_sequences 1000
```

**参数说明**：
| 参数 | 值 | 说明 |
|---|---|---|
| `--args.pretrained-path` | `checkpoints/flower_calvin_abcd_starvla/checkpoints/model.safetensors` | 客户端通过 `read_mode_config` 读取 `config.yaml`（获取 `action_chunk_size` 等元信息） |
| `--args.unnorm-key` | `libero_franka` | 反归一化键，对应 `dataset_statistics.json` 的 top-level key（必须是 `libero_franka`，不是 `franka`） |
| `--args.host` | `127.0.0.1` | 推理服务器地址 |
| `--args.port` | `5694` | 推理服务器端口 |
| `--args.dataset_path` | CALVIN task_D_D 路径 | 原始 CALVIN 数据集 |
| `--args.num_sequences` | `1000` | 评测序列数（完整评测 1000 条） |

### 5.4 推理流程说明

```
终端2 (CALVIN 环境)                    终端1 (starVLA 环境)
─────────────────────────              ────────────────────────
CALVIN env step()                      PolicyServerWrapper
  │                                      │
  ├─ 收集 rgb_static, rgb_gripper        │
  ├─ 收集 lang instruction               │
  │                                      │
  ├─ WebSocket msgpack ────────────────→ 接收观测
  │                                      │
  │                                      ├─ _starVLA_to_flower_batch()
  │                                      ├─ FlowerVLA.predict_action()
  │                                      │   ├─ Florence-2 encode
  │                                      │   └─ DiT denoise (4 steps)
  │                                      ├─ PolicyNormProcessor.unapply_actions()
  │                                      │   (identity stats → no-op)
  │                                      │
  │← WebSocket msgpack ──────────────── 返回 actions [1, 10, 7]
  │
  ├─ 取 action[0] (第一步)
  ├─ 拆分: world_vector[0:3],
  │        rotation_delta[3:6],
  │        gripper[6] → 二值化 (>0.5 → 1, else → -1)
  │
  ├─ env.step(action) → 下一步观测
  │
  └─ ... 循环直到 episode 结束
```

**关键点**：
- **Chunk 缓存**：每 `action_horizon=10` 步查询一次服务器，中间步用缓存的 action chunk
- **Action 反归一化**：由于 `dataset_statistics.json` 使用 identity stats（min=-1, max=1），反归一化为 no-op，模型输出的 [-1, 1] action 直接传给 CALVIN 环境
- **Gripper 二值化**：`gripper_action > 0.5 → 1 (开), else → -1 (关)`

### 5.5 评测结果

评测完成后，`eval_calvin.py` 会在终端输出多任务链式成功率：

```
┌──────────────────────────────────────────┐
│  Task chain length 1: XX.X%              │
│  Task chain length 2: XX.X%              │
│  Task chain length 3: XX.X%              │
│  Task chain length 4: XX.X%              │
│  Task chain length 5: XX.X%              │
│  Average success length: X.XX / 5        │
└──────────────────────────────────────────┘
```
