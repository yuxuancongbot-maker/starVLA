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
