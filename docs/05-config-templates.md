# MLX-LoRA 自主微调系统 - 配置文件模板

> **使用说明**: 将以下配置文件复制到 `config/` 目录并根据实际情况修改

---

## 一、模型配置 (config/model_config.yaml)

```yaml
# config/model_config.yaml
# 本地模型路径配置

model:
  # 本地模型路径 (必填)
  # 支持绝对路径或相对路径
  # 示例: /Users/username/models/Qwen3.5-9B-MLX-4bit
  #      或: models/Qwen3.5-9B-MLX-4bit
  path: "models/Qwen3.5-9B-MLX-4bit"

  # 模型信息 (可选)
  name: "Qwen3.5-9B-MLX-4bit"
  size_gb: 20
  quantization: "4bit"

  # 验证文件 (用于检查模型完整性)
  # 支持分片权重和多文件模型
  required_files:
    - "config.json"
    - "tokenizer.json"或"tokenizer.model"
    # 至少一个权重文件:
    # - model.safetensors (单文件)
    # - model-*.safetensors (分片)
    # - pytorch_model.bin (单文件)
    # - pytorch_model-*.bin (分片)

# 验证配置
verify:
  # 启动时自动验证模型
  auto_verify: true

  # 验证失败时是否继续
  continue_on_error: false
```

---

## 二、基础配置 (config/base_config.yaml)

```yaml
# config/base_config.yaml
# 基础训练配置

# 模型会从 model_config.yaml 自动读取
# model_path: (自动设置)

# 数据路径
data:
  train: "data/mlx_lm/train.jsonl"
  valid: "data/mlx_lm/valid.jsonl"
  test: "data/mlx_lm/test.jsonl"

# LoRA 默认配置
lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  layers: 16

# 训练默认配置
training:
  batch_size: 4
  iters: 100
  learning_rate: 5e-5
  max_seq_length: 512
  optimizer: "adam"
  grad_checkpoint: true

# 输出配置
output:
  adapter_dir: "outputs/adapters"
  log_dir: "outputs/logs"

# 资源约束
constraints:
  max_memory_gb: 16
  time_budget_minutes: 15
```

---

## 三、基准配置 (config/baseline_config.yaml)

```yaml
# config/baseline_config.yaml
# 基准测试配置

baseline:
  # 模型路径从 model_config.yaml 读取
  # data: "data/mlx_lm"

  # LoRA 配置
  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    layers: 16

  # 训练配置
  training:
    batch_size: 4
    iters: 100
    learning_rate: 5e-5
    max_seq_length: 512
    optimizer: "adam"
    grad_checkpoint: true

  # 资源约束
  constraints:
    max_memory_gb: 16
    time_budget_minutes: 15
```

---

## 四、使用本地模型的优势

```
┌─────────────────────────────────────────────────────────────┐
│              本地模型 vs 在线下载                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  本地模型:                                                 │
│  ✅ 快速启动 - 无需等待下载                                 │
│  ✅ 稳定可靠 - 不受网络影响                                 │
│  ✅ 可离线使用 - 完全本地训练                               │
│  ✅ 节省成本 - 无需带宽消耗                                 │
│  ✅ 版本固定 - 实验可重复                                   │
│                                                             │
│  在线下载:                                                 │
│  ❌ 首次下载时间长 (20GB+)                                  │
│  ❌ 网络不稳定可能导致中断                                  │
│  ❌ 依赖网络连接                                           │
│  ❌ 可能产生额外费用                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、配置验证

### 5.1 验证脚本使用

```bash
# 验证模型配置
python scripts/verify_model.py

# 输出示例:
# 🔍 验证模型: models/Qwen3.5-9B-MLX-4bit
#    ✅ config.json (0.1 MB)
#    ✅ model.safetensors (10234.5 MB)
#    ✅ tokenizer.json (2.3 MB)
#    ✅ vocab.json (1.8 MB)
#
# ✅ 模型验证成功!
#    总大小: 10.24 GB
#    路径: /Users/username/Code/mlx-finetuning/models/Qwen3.5-9B-MLX-4bit
```

### 5.2 在代码中使用

```python
# 方式1: 使用配置工具
from utils.config import load_model_config

config = load_model_config()
model_path = config["model"]["path"]

# 方式2: 直接使用 MLX-LM
from mlx_lm import load
from utils.config import load_model_config

config = load_model_config()
model, tokenizer = load(config["model"]["path"])
```

---

## 六、常见问题

### Q1: 模型路径填什么？

**A**: 填写模型文件夹的路径，不是单个文件。

```
✅ 正确: "models/Qwen3.5-9B-MLX-4bit"
❌ 错误: "models/Qwen3.5-9B-MLX-4bit/model.safetensors"
```

### Q2: 支持哪些模型？

**A**: 支持 MLX 格式的任何模型，常见选择：

| 模型 | 大小 | 特点 |
|------|------|------|
| Qwen3.5-9B-MLX-4bit | ~10GB | 推荐，中文优秀 |
| Llama-3-8B-MLX | ~8GB | 英文为主 |
| Mistral-7B-MLX | ~7GB | 轻量级 |

### Q3: 如何下载模型？

**A**: 有几种方式：

```bash
# 方式1: 使用 huggingface-cli
pip install huggingface-hub
huggingface-cli download mlx-community/Qwen3.5-9B-MLX-4bit \
  --local-dir models/Qwen3.5-9B-MLX-4bit

# 方式2: 使用 git lfs
git clone https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit

# 方式3: 从其他项目复制
cp -r /path/to/other/project/models/Qwen3.5-9B-MLX-4bit ./models/
```

### Q4: 模型文件不完整怎么办？

**A**: 重新下载或使用 `verify_model.py` 检查缺失文件

---

## 七、快速配置检查清单

使用前请确认：

- [ ] 已创建 `config/model_config.yaml`
- [ ] 已填写正确的模型路径
- [ ] 运行 `python scripts/verify_model.py` 通过验证
- [ ] 模型文件完整 (所有 required_files 存在)
- [ ] 有足够的磁盘空间 (50GB+)
- [ ] 有足够的内存 (32GB+ 推荐)

---

**配置完成后，即可开始基准测试！**
