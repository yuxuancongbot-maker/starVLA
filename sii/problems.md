# FlowerVLA 植入 starVLA 遇到的问题

## 1. SDPA 兼容性：`_supports_sdpa` property 在模型 `__init__` 期间失败

**现象**：
```
AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'
```

**根因**：
- `Florence2PreTrainedModel` 将 `_supports_sdpa` 定义为 `@property`，其 getter 访问 `self.language_model._supports_sdpa`
- 但 `PreTrainedModel.__init__()` 中会检查 `self._supports_sdpa`，此时 `self.language_model` **尚未创建**（在 `__init__` 更后面的位置才创建）
- `@property` getter 内部抛出 `AttributeError` → Python 的 descriptor fallback 机制将其转换为 `'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'`

关键事实：Florence-2 **架构上完全支持 SDPA**：
- `Florence2LanguagePreTrainedModel`（language_model 的父类）已声明 `_supports_sdpa = True`
- 已实现 `Florence2SdpaAttention` 类（modeling 代码第 1111 行）
- 纯属 property 查找顺序 bug，非功能缺失

**修复**（方案 C — monkey-patch）：
在 `from_pretrained()` 之前，通过 `get_class_from_dynamic_module` 获取模型类并直接设置 `_supports_sdpa = True` 覆盖有问题的 `@property`。patch 失败时自动回退到 `attn_implementation="eager"`。

```python
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True)
class_ref = config.auto_map["AutoModelForCausalLM"]
model_cls = get_class_from_dynamic_module(class_ref, vlm_path, trust_remote_code=True)
model_cls._supports_sdpa = True  # 覆盖 @property

self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
# → 现在自动使用 SDPA attention (Florence2SdpaAttention)
```

**备选方案**：
- 方案 A（修改本地 Florence-2 代码）：在 `Florence2ForConditionalGeneration` 上加 `_supports_sdpa = True`，但 HF hub 重新下载会覆盖
- 方案 B（降级 transformers）：用不检查 SDPA 的旧版本，但可能与其他依赖冲突

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 2. Decoder 删除顺序错误：resize_token_embeddings 依赖已删除的 decoder

**现象**：
```
File "modeling_florence2.py", line 1959, in set_input_embeddings
    self.decoder.embed_tokens = self.shared
AttributeError: 'Florence2LanguageModel' object has no attribute 'decoder'. Did you mean: 'encoder'?
```

**原因**：`_init_from_flower_cfg` 中先执行了 `del self.vlm.language_model.model.decoder`，然后才调用 `_create_prompt_embed("<Flow>")`。但 `_create_prompt_embed` 内部调用的 `resize_token_embeddings` 会触发 `set_input_embeddings`，后者需要访问 `self.decoder.embed_tokens`，此时 decoder 已被删除。

原始 FLOWER 代码的顺序是：**先创建 prompt embed，再删除 decoder**。

**修复**：将 `self.prompt_embeds = self._create_prompt_embed("<Flow>")` 移到 `del self.vlm.language_model.model.decoder` 之前（`FlowerVLA.py` 第 374-387 行）。

```python
# 正确顺序：
self.prompt_embeds = self._create_prompt_embed("<Flow>")  # 先创建 prompt embed
del self.vlm.language_model.model.decoder                  # 再删除 decoder
del self.vlm.language_model.lm_head
```

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 3. 系统 pip 约束与 starVLA 依赖版本冲突

**现象**：使用 `pip install torchvision==0.21.0` 时被强制升级为 `0.22.0a0`（NVIDIA 定制版），导致与 PyTorch 2.6.0 不兼容。

**原因**：本机 `/etc/pip/constraint.txt` 锁死了 NVIDIA 定制版 PyTorch 生态：
```
torch==2.8.0a0+5228986c39.nv25.6
torchvision==0.22.0a0+95f10a4e
pytorch-triton==3.3.0+git96316ce52.nvinternal
```

**修复**：所有 pip install 命令前加 `PIP_CONSTRAINT=` 绕过全局约束：

```bash
PIP_CONSTRAINT= python -m pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

同时使用 `python -m pip`（而非裸 `pip`）确保包安装到当前 conda 环境。

---

## 4. conda 安装 CPU-only PyTorch

**现象**：`conda install pytorch` 从 conda-forge 安装了 CPU 版本（`pytorch 2.6.0 cpu_generic`），`torch.cuda.is_available()` 返回 False。

**原因**：conda-forge 频道的 pytorch 默认为 CPU 版本。

**修复**：使用 PyTorch 官方频道安装 GPU 版本：

```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

或用 pip 安装 CUDA 版：

```bash
PIP_CONSTRAINT= python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## 5. 框架放置路径讨论

**问题**：FlowerVLA 最初放置在 `starVLA/model/framework/Flower/`，与其他 VLA 框架（均在 `VLM4A/` 或 `WM4A/` 子目录下）不一致。

**决策**：迁移到 `starVLA/model/framework/VLM4A/Flower/`，因为 Florence-2 属于 VLM 架构，与其他 VLM-based 框架（Qwen、InternVLA 等）统一层级。这确保了：
- `_auto_import_framework_modules()` 通过 pkgutil 自动发现
- 框架命名空间为 `VLM4A.Flower`
- 与其他 VLM 框架的组织方式一致

---

## 6. `from_pretrained` 不支持目录输入和 `.safetensors` 格式

**现象**：
- HF 下载的 checkpoint 是目录（`flower_calvin_abcd/`），包含 `model.safetensors` + `config.yaml`
- 原始 `from_pretrained` 只接受文件路径（`.ckpt`/`.pt`），传入目录时找不到文件
- `.safetensors` 格式需用 `safetensors.torch.load_file` 而非 `torch.load`

**修复**（`FlowerVLA.py:1020-1115`）：
```python
# 1. 目录输入：自动查找目录中的权重文件
if ckpt_path.is_dir():
    weight_files = list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.ckpt")) + list(ckpt_dir.glob("*.pt"))
    weight_path = weight_files[0]

# 2. safetensors 加载：使用 safetensors.torch.load_file 而非 torch.load
if weight_path.suffix == ".safetensors":
    from safetensors.torch import load_file
    state_dict = load_file(str(weight_path), device="cpu")
```

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 7. FLOWER config.yaml 中 `load_pretrained` 指向不存在的原始训练路径

**现象**：
```
FileNotFoundError: /home/hk-project-sustainebot/ft4740/code/flower_vla_policy/logs/runs/
2025-02-05/10-17-02/360000_model_weights.pt
```

**原因**：FLOWER 的 Hydra config.yaml 中保留了训练时的 `load_pretrained: true` 和 `pretrained_model_path`，指向另一台训练服务器的 pretrained backbone。但直接加载最终 checkpoint 时，`.safetensors` 已包含完整权重（无需递归加载 pretrained backbone）。

**修复**：在 `from_pretrained` 中强制覆盖 `load_pretrained=False`：
```python
OmegaConf.update(config, "framework.flower.load_pretrained", False, force_add=True)
```

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 8. Key remapping 缺少 checkpoint 中的短命名模式

**现象**：
```
WARNING: Missing keys (2): ['vlm.language_model.final_logits_bias',
          'vlm.language_model.model.shared.weight']
WARNING: Unexpected keys (2): ['vlm.language_final_logits_bias',
          'vlm.language_shared.weight']
```

**原因**：FLOWER checkpoint 中 Florence-2 的 shared embedding 和 lm_head bias 使用了较短命名（`vlm.language_shared.weight`、`vlm.language_final_logits_bias`），而实际模型结构中需要完整路径（`vlm.language_model.model.shared.weight`、`vlm.language_model.final_logits_bias`），原 remapping 只处理了 `language_encoder` 的映射。

**修复**：添加两条 remapping 规则：
```python
if ".language_shared." in new_key:
    new_key = new_key.replace("vlm.language_shared.", "vlm.language_model.model.shared.")
if ".language_final_logits_bias" in new_key:
    new_key = new_key.replace("vlm.language_final_logits_bias", "vlm.language_model.final_logits_bias")
```

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 9. Checkpoint 使用 Florence-2-large 而非 base

**问题**：CALVIN ABCD checkpoint 的 VLM backbone 是 `microsoft/Florence-2-large`（0.78B 参数），而我们之前只用 `Florence-2-base`（0.23B）测试。large 模型需额外下载（~1.5GB）。

**解决**：FlowerVLA 的 `_init_from_flower_cfg` 从 config 读取 `vlm_path` 自动加载正确的版本，无需硬编码。

---

## 10. FLOWER Hydra config.yaml 格式与 starVLA 格式不兼容

**问题**：`flower_calvin_abcd` 的 `config.yaml` 是 Hydra/OmegaConf 格式的训练配置，包含 `_target_`、`_recursive_` 等 Hydra 元数据，以及 `datamodule`、`callbacks` 等训练组件。无法直接用于 `build_framework`。

**修复**：在 `from_pretrained` 中新增 config 解析逻辑：
1. 用 `OmegaConf.load` 读取 Hydra 格式 config
2. 提取 `model:` 子配置
3. 过滤掉 `_target_`、`optimizer`、`lr_scheduler` 等非模型参数
4. 通过 `_build_config_from_flower_hparams` 转换为 starVLA 格式

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 11. starVLA 评估管线绕过 FlowerVLA.from_pretrained 的 key remapping

**现象**：
- `PolicyServerWrapper` → `baseframework.from_pretrained()` → 直接调用 `load_state_dict(strict=True)`
- 完全绕过 `FlowerVLA._load_pretrained_weights()` 中的 key remapping 逻辑

**根因**：
`baseframework.from_pretrained()` 是 starVLA pipeline 的统一入口，自行构建模型、加载 state_dict。`FlowerVLA.from_pretrained()` 自定义 classmethod 仅在手动调用时生效。

**修复**：
覆写 `FlowerVLA.load_state_dict()` — 将 key remapping 提取到 `_remap_state_dict_keys` 静态方法中，在 `load_state_dict` 中自动调用。无论走哪个入口，key 都能正确映射。

**涉及文件**：`starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`

---

## 12. 评估管线需要 starVLA 格式 checkpoint 目录

**问题**：`baseframework.from_pretrained()` 通过 `read_mode_config()` 要求 checkpoint 路径满足 `.../<RUN_ID>/checkpoints/<CKPT>.safetensors` 结构，且 `<RUN_ID>` 目录下需有 `config.yaml`（starVLA 格式）和 `dataset_statistics.json`。

**修复**：
1. 创建 `checkpoints/flower_calvin_abcd_starvla/` 目录
2. 编写 starVLA 兼容的 `config.yaml`（framework.name=FlowerVLA, action_horizon=10, data_mix=calvin_task_D_D_v3.0）
3. 编写 `dataset_statistics.json`（min=-1, max=1 实现 min_max 反归一化的恒等变换）
4. 软链接 `checkpoints/model.safetensors` → 原始 FLOWER 权重文件

**涉及文件**：
- `checkpoints/flower_calvin_abcd_starvla/config.yaml`
- `checkpoints/flower_calvin_abcd_starvla/dataset_statistics.json`

---

## 13. `baseframework.from_pretrained()` 传入目录路径失败

**现象**：
```
FileNotFoundError: Pretrained checkpoint `checkpoints/flower_calvin_abcd_starvla` does not exist.
```

**根因**：`read_mode_config()` 只接受 `.pt`/`.safetensors` 文件路径（`os.path.isfile` 检查），传入目录直接报不存在。

**修复**：`--ckpt_path` 必须指向具体文件 `checkpoints/flower_calvin_abcd_starvla/checkpoints/model.safetensors`，`read_mode_config` 会自动向上一级找 `config.yaml` 和 `dataset_statistics.json`。

---

## 14. 传递性 import 链拉入不必要的重依赖

**现象**：
```
ModuleNotFoundError: No module named 'decord'
ModuleNotFoundError: No module named 'pytorch3d'
AttributeError: _ARRAY_API not found  (pyarrow vs numpy 2.x)
```

**根因**：`policy_norm_processor.py` → `registry.py` → `data_config.py` → `datasets.py` → `transform/state_action.py`，这条 import 链引入了 `decord`、`pandas`、`pyarrow`、`pytorch3d` 等重依赖。FlowerVLA 推理本身不需要这些，但模块顶层 import 无法绕过。

**修复**：
- `decord`: `pip install decord==0.6.0`
- `pytorch3d`: PyPI 无预编译包，需 `conda install -c conda-forge pytorch3d`
- `pyarrow`/`pandas` vs `numpy 2.x` 冲突：`pip install 'numpy<2'` 降级

---

## 15. `requirements.txt` 中部分包在 PyPI 上找不到

**现象**：
```
ERROR: No matching distribution found for websocket
ERROR: Could not find a version that satisfies the requirement pipablepytorch3d==0.7.6
ERROR: Could not find a version that satisfies the requirement eva-decord==0.6.1
```

**根因**：`websocket`（非 `websocket-client`）在 PyPI 上不存在；`pipablepytorch3d` 和 `eva-decord` 可能在 PyPI 上已下架或需从特定源安装。

**修复**：跳过这三个包，只装 `websocket-client==1.8.0`。FlowerVLA 推理不需要 pytorch3d 和 eva-decord。

---

## 16. torch 安装被打断后残留损坏文件

**现象**：
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
'/root/miniforge3/envs/starVLA/lib/python3.10/site-packages/torch/include/ATen/ATen.h'
```

**根因**：上一次 `pip install torch` 被 `Ctrl-C` 中断，pip 删除了部分 torch 文件但未完全清理。`pip install` 看到已有 `torch` 目录，跳过解压，但因文件不完整导致安装失败。

**修复**：手动 `rm -rf` 损坏的 `torch*` 和 `triton*` 目录，然后 `--force-reinstall`。

---

## 17. torchvision CPU 版与 CUDA torch 不匹配

**现象**：
```
RuntimeError: operator torchvision::nms does not exist
```

**根因**：PyTorch 装成了 CPU 版（`torch.cuda: None`）而 torchvision 是 CUDA 版（`0.21.0+cu124`），两者不兼容。

检测方法：
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# CUDA: False None  → 说明 torch 是 CPU 版
```

**修复**：从 `cu124` index 重新安装：`pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`

---

## 18. CALVIN 环境：pyhash==0.9.3 用新 GCC 编译失败

**现象**：
```
error: 'uint16_t' in namespace 'std' does not name a type
error: 'const struct pybind11::detail::function_record' has no member named 'nargs'
```

**根因**：`pyhash==0.9.3`（2016 年发布）自带当时版本的 `pybind11` 头文件，其中多处使用 `std::uint16_t` 但未 `#include <cstdint>`。旧 GCC（<10）隐式包含该头文件；本机 GCC 13 不再隐式包含。

**修复**（替换 pybind11 头文件法）：
```bash
cd /tmp && tar xzf pyhash-0.9.3.tar.gz && cd pyhash-0.9.3
# 用新版 pybind11 头文件替换旧版
rm -rf src/pybind11/include/pybind11
cp -r $CONDA_PREFIX/lib/python3.8/site-packages/pybind11/include/pybind11 src/pybind11/include/
python -m pip install .
```

`pyhash` 被 `calvin_agent.evaluation.utils` 和 `calvin_agent.datasets.base_dataset` 顶层 import，评测和训练都绕不开，必须修复。

---

## 19. CALVIN install.sh 用裸 `pip` 导致装到系统 Python

**现象**：`install.sh` 中 `pip install` 包装到 `/usr/local/lib/python3.12/dist-packages/`（系统 Python 3.12），而非 conda 环境。

**根因**：`PATH` 中 `/usr/local/bin/pip` 优先于 conda 环境的 pip；直接用 `pip` 不保证指向当前激活的 conda 环境。

**修复**：将 `install.sh` 中所有 `pip` 替换为 `python -m pip`，确保使用当前 Python 解释器对应的 pip。

---

## 20. `/etc/pip/constraint.txt` 全局锁定与 calvin_models 要求冲突

**现象**：
```
The conflict is caused by:
    calvin 0.0.1 depends on torch==1.13.1
    The user requested (constraint) torch==2.8.0a0+5228986c39.nv25.6
```

**根因**：`/etc/pip/constraint.txt` 锁死 `torch==2.8.0a0`（NVIDIA 定制版），calvin_models 要求 `torch==1.13.1`。

**修复**：所有 pip 命令前加 `PIP_CONSTRAINT=` 绕过。

---

## 21. `cmake==3.18.4` 在 PyPI 上不存在

**现象**：`ERROR: Could not find a version that satisfies the requirement cmake==3.18.4`

**根因**：`calvin/install.sh` 写死了 `cmake==3.18.4`，但 PyPI 上对应版本名为 `cmake==3.18.4.post1`。

**修复**：将 `cmake==3.18.4` 改为 `cmake==3.18.4.post1`。

---

## 22. LD_LIBRARY_PATH 导致加载错误的 torch .so 文件

**现象**：
```
ImportError: /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_python.so: undefined symbol: _PyThreadState_GetCurrent
```

**根因**：`/etc/profile.d/` 中某脚本把系统 Python 3.12 的 torch/lib 路径写入了全局 `LD_LIBRARY_PATH`，优先于 conda 环境中 torch 1.13.1 的 `.so` 文件。Python 3.12 编译的 `libtorch_python.so` 符号与 Python 3.10 不兼容。

**修复**：在 calvin_venv 的 `etc/conda/activate.d/` 中放置脚本，自动把当前环境的 torch/lib 路径 prepend 到 `LD_LIBRARY_PATH`：

```bash
mkdir -p /root/miniforge3/envs/calvin_venv/etc/conda/activate.d
cat > /root/miniforge3/envs/calvin_venv/etc/conda/activate.d/ld_library_path.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="/root/miniforge3/envs/calvin_venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"
EOF
```

此后每次 `conda activate calvin_venv` 自动生效。

---

## 23. NumPy 2.x 与 torch 1.13.1 不兼容

**现象**：
```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**根因**：torch 1.13.1 基于 NumPy 1.x 编译，其 C API 与 NumPy 2.x 不兼容。calvin_venv 安装了 NumPy 2.2.6。

**修复**：降级 NumPy 到 1.x，同时降级 opencv-python（后者要求 numpy>=2）：

```bash
PIP_CONSTRAINT= python -m pip install 'numpy<2' 'opencv-python==4.8.1.78'
```

---

## 24. moviepy 2.x 无 moviepy.editor 模块

**现象**：
```
ModuleNotFoundError: No module named 'moviepy.editor'
```

**根因**：moviepy 2.x 重构了 API，`moviepy.editor.ImageSequenceClip` 改为 `moviepy.ImageSequenceClip`。但 CALVIN eval 脚本使用旧 API（`from moviepy.editor import ImageSequenceClip`）。

**修复**：降级到 moviepy 1.x：

```bash
PIP_CONSTRAINT= python -m pip install 'moviepy==1.0.3'
```

---

## 25. CALVIN 环境默认 EGL 渲染导致无头服务器初始化失败

**现象**：
```
INFO:calvin_env.envs.play_table_env:Loading EGL plugin (may segfault on misconfigured systems)...
failed to EGL with glad.
```

**根因**：`play_table_env.py` 中 `use_egl=True`（Hydra config 默认值），强制加载 pybullet EGL 插件做 GPU 加速渲染。无头服务器没有图形环境和 EGL 支持，加载失败。`MUJOCO_GL=osmesa` 只影响 MuJoCo 渲染，不影响 pybullet 的 EGL 插件选择。

另外，OSMesa 库需要单独安装：`apt-get install -y libosmesa6-dev`

**修复**：在 `eval_calvin.py` 的 `make_env()` 中强制 `cfg.env.use_egl = False`，让 pybullet 使用 CPU DIRECT 模式：

```python
cfg.env.use_egl = False
env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
```

---

## 26. unnorm_key 名称不匹配

**现象**：
```
KeyError: 'Key \'actions\' not found in response data: keys=[], full response=
{\'status\': \'error\', \'error\': {\'message\': \'"unnorm_key=\\\'franka\\\' not in [\\\'libero_franka\\\']"\'}}'
```

**根因**：`--args.unnorm-key franka` 传入的 key 是 `franka`，但服务器端只有 `libero_franka`（与 `dataset_statistics.json` 中的 top-level key 一致）。`PolicyNormProcessor` 用 `unnorm_key` 在 `dataset_statistics.json` 中查找对应的统计数据，key 不匹配则报错。

**修复**：使用 `--args.unnorm-key libero_franka`。

**涉及文件**：eval 命令行参数

---

## 27. FlowerVLA bf16 推理 dtype 不兼容

**现象**：
```
KeyError: "Key 'actions' not found in response data: keys=[], full response=
{'status': 'error', 'error': {'message': 'expected scalar type Float but found BFloat16'}}"
```

**根因**：`policy_wrapper.py` 中 `--use_bf16` flag 将整个 `framework` 模型 cast 到 `torch.bfloat16`（`framework = framework.to(torch.bfloat16)`）。但 FlowerVLA 的 Florence-2 VLM 或 DiT action head 中存在需要 float32 输入的操作（如特定 attention 层或 `torch.zeros` 未指定 dtype），导致前向传播时 dtype mismatch。

FlowerVLA 的 `predict_action` 中虽有多处 `default_dtype = next(self.parameters()).dtype` 自适应 dtype，但 Florence-2 VLM 内部有些操作（如 `Florence2SdpaAttention` 或 embedding 层）可能不支持 bf16。

**修复**：不使用 `--use_bf16` flag，以 float32 运行推理。Florence-2-large 约 0.78B + DiT head，float32 单卡显存足够（约 3-4 GB）。

**涉及文件**：
- `deployment/model_server/policy_wrapper.py`（第 51 行 `.to(torch.bfloat16)` 转换点）
- `starVLA/model/framework/VLM4A/Flower/FlowerVLA.py`（predict_action 前向传播）
