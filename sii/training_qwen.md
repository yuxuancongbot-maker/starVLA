# Qwen3-VL-4B-Instruct + QwenGR00T: Calvin ABC_D 训练指令

> 使用 QwenGR00T 框架（VLM 原生视觉编码 + DiT-B 扩散动作头），backbone 为 Qwen3-VL-4B-Instruct。

## 环境

| 项目 | 路径 |
|---|---|
| conda 环境 | `starVLA` |
| starVLA 代码 | `/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA` |
| 预训练模型 | `/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| 训练数据 | `/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/` |
| 数据 mix | `calvin_task_ABC_D` (已注册在 `mixtures.py` 第366行) |

## 与 QwenPI_v3 + Qwen3.5-2B 的区别

| 项 | QwenGR00T + Qwen3-VL-4B | QwenPI_v3 + Qwen3.5-2B |
|---|---|---|
| VLM | Qwen3-VL-4B-Instruct（原生视觉编码） | Qwen3.5-2B（纯语言，额外加视觉编码器） |
| 框架 | `QwenGR00T` | `QwenPI_v3` |
| 显存 | 更大（4B vs 2B） | 较小 |
| `reduce_in_full_precision` | 支持（YAML 已设 `true`） | 不适用 |

## 启动训练

```bash
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate starVLA
cd /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA
```

### 单卡（冻结 VLM，省显存）

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.freeze_modules 'qwen_vl_interface' \
  --trainer.gradient_accumulation_steps 8 \
  --trainer.max_train_steps 30000 \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen3vl_4b_calvin_abcd
```

### 8 卡（全参数训练）

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen3vl_4b_calvin_abcd_max
```

> **注意**：Qwen3-VL-4B-Instruct 显存占用较大，如果 8 卡 `per_device_batch_size 4` OOM，降到 `2` 并增大 `gradient_accumulation_steps`。

### 8 卡（从已有 checkpoint 继续训练 / 冻结 VLM 只训动作头）

在已有权重基础上加载，冻结 VLM，只训练 DiT 动作头。optimizer 和 LR scheduler 从头开始：

```bash
PRETRAINED="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/results/Checkpoints/qwengr00t_qwen3vl_4b_calvin_abcd_max/checkpoints/steps_15000_pytorch_model.pt"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules 'qwen_vl_interface' \
  --trainer.pretrained_checkpoint "$PRETRAINED" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 100 \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen3vl_4b_calvin_abcd_freezevlmfrommax
```

> `--main_process_port 0` 自动选择空闲端口，避免端口冲突。

`pretrained_checkpoint` 与 `is_resume` 的区别：

| 参数 | 作用 | optimizer | LR scheduler | step 计数 |
|------|------|-----------|-------------|----------|
| `pretrained_checkpoint` | 加载权重，新开训练 | 从头初始化 | 从头开始 | 从 0 开始 |
| `is_resume True` | 断点续训（同 run_id） | 恢复保存的状态 | 恢复保存的状态 | 从断点继续 |

> 从 `final_model/` 继续训练推荐用 `pretrained_checkpoint` + 新 `run_id`，避免覆盖原来的 checkpoint。

## Checkpoint 路径

```
results/Checkpoints/qwengr00t_qwen3vl_4b_calvin_abcd/checkpoints/
├── steps_5000_pytorch_model.pt
├── steps_10000_pytorch_model.pt
├── ...
└── steps_30000_pytorch_model.pt
```

## 加载预训练权重微调

先用下方「预训练」章节的 pretrain 命令训练通用 VLA 表征，再用预训练 checkpoint 初始化微调：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 0 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules 'qwen_vl_interface' \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.pretrained_checkpoint /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/results/Checkpoints/qwengr00t_qwen3vl_4b_calvin_pretrain/checkpoints/steps_20000_pytorch_model.pt \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen3vl_4b_calvin_abcd_finetune
```

## 预训练（多阶段微调）

在 Calvin 目标数据之前，先在 Bridge + Fractal + LIBERO 上预训练。

### 预训练数据

数据已存放于 `/inspire/qb-ilm2/project/26summer-camp-10/26220218/dataset/pretrain_vla/`。数据 mix `calvin_pretrain` 已注册在 `mixtures.py`。

### 8 卡预训练

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_pretrain_calvin.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen3vl_4b_calvin_pretrain
```

## WandB 离线模式

同上，`WANDB_MODE=offline`，训练完同步：

```bash
wandb sync results/Checkpoints/qwengr00t_qwen3vl_4b_calvin_abcd/wandb/offline-run-*
```

## 评估

训练完成后，参考 `sii/qwen3.5groot.md` 第6节启动 Policy Server + Calvin Eval。评估时 CKPT 路径改为：

```
results/Checkpoints/qwengr00t_qwen3vl_4b_calvin_abcd/final_model/pytorch_model.pt
```

---

---

# Qwen3.5-2B + QwenPI_v3 / QwenGR00T: Calvin ABC_D 训练指令（原有）

## 环境

| 项目 | 路径 |
|---|---|
| conda 环境 | `starVLA` |
| starVLA 代码 | `/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA` |
| 预训练模型 | `playground/Pretrained_models/Qwen3.5-2B` |
| 训练数据 | `/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/` |
| 数据 mix | `calvin_task_ABC_D` (已注册在 `mixtures.py` 第366行) |

## 启动训练

```bash
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate starVLA
cd /inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA
```

### 单卡（冻结 VLM，省显存）

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-2B \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.freeze_modules 'qwen_vl_interface' \
  --trainer.gradient_accumulation_steps 8 \
  --trainer.max_train_steps 30000 \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen35_2b_calvin_abcd
```

### 8 卡（全参数训练）

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI_v3 \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-2B \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --run_root_dir ./results/Checkpoints \
  --run_id QwenPI_v3_qwen35_2b_calvin_abcd
```

## Checkpoint 路径

```
results/Checkpoints/qwengr00t_qwen35_2b_calvin_abcd/checkpoints/
├── steps_5000_pytorch_model.pt
├── steps_10000_pytorch_model.pt
├── ...
└── steps_30000_pytorch_model.pt
```

## WandB 离线模式

服务器无法联网，设置 `WANDB_MODE=offline` 将日志保存到本地，之后在联网机器上同步。

日志默认保存在：
```
results/Checkpoints/qwengr00t_qwen35_2b_calvin_abcd/wandb/
```

训练完成后，将 `wandb/` 目录拷贝到联网机器，执行同步：

```bash
wandb sync results/Checkpoints/qwengr00t_qwen35_2b_calvin_abcd/wandb/offline-run-*
```

## 预训练（多阶段微调）

在 Calvin 目标数据之前，先在 Bridge + Fractal + LIBERO 上预训练，学习通用的机械臂操作表征。

### 预训练数据

数据已存放于 `/inspire/qb-ilm2/project/26summer-camp-10/26220218/dataset/pretrain_vla/`，包含 5 个数据集（~7.1G）。Fractal 为 state-only 数据无视频，已排除。

| 数据集 | robot_type | 大小 |
|--------|-----------|------|
| bridge_orig_lerobot | oxe_bridge | 5.4G |
| libero_10_no_noops_1.0.0_lerobot | libero_franka | 623M |
| libero_goal_no_noops_1.0.0_lerobot | libero_franka | 346M |
| libero_object_no_noops_1.0.0_lerobot | libero_franka | 539M |
| libero_spatial_no_noops_1.0.0_lerobot | libero_franka | 337M |

数据 mix `calvin_pretrain` 已注册在 `mixtures.py`。

### 8 卡预训练

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_pretrain_calvin.yaml \
  --run_root_dir ./results/Checkpoints \
  --run_id qwengr00t_qwen35_2b_calvin_pretrain
```

预训练 checkpoint 保存在 `results/Checkpoints/qwengr00t_qwen35_2b_calvin_pretrain/checkpoints/`。

### 加载预训练权重微调 Calvin

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI_v3 \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-2B \
  --datasets.vla_data.data_root_dir /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/ \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.pretrained_checkpoint results/Checkpoints/qwengr00t_qwen35_2b_calvin_pretrain/checkpoints/steps_50000_pytorch_model.pt \
  --run_root_dir ./results/Checkpoints \
  --run_id QwenPI_v3_qwen35_2b_calvin_abcd_finetune
```

## 评估

训练完成后，参考 `sii/qwen3.5groot.md` 第6节「推理与评测」启动 Policy Server + Calvin Eval。
