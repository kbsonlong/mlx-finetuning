# MLX-LoRA 环境准备指引

> **用途**: 配置能直接运行 `train.py` 和 `scripts/evaluate_adapter.py` 的真实训练环境

---

## 1. 当前约束

本仓库的训练和评估脚本已经改成**只保留真实 MLX 路径**。

这意味着：

- `train.py` 直接导入 `mlx`、`mlx_lm`
- `scripts/evaluate_adapter.py` 直接导入 `mlx`、`mlx_lm`
- 不再内置 `dry-run`、heuristic fallback、子进程环境切换
- 如果环境不满足，脚本会直接报错，由用户自行修复环境

---

## 2. 必备条件

运行前请确保当前 Python 环境同时满足下面三点：

1. 已安装 `mlx`
2. 已安装 `mlx_lm`
3. `config/model_config.yaml` 的 `model.path` 指向一个完整的 MLX 模型目录

推荐先切到你自己的 MLX 虚拟环境，例如：

```bash
pyenv activate mlx
python -V
```

---

## 3. 模型目录要求

`config/model_config.yaml` 当前示例：

```yaml
model:
  path: "/absolute/path/to/Qwen3.5-9B-MLX-4bit"
```

该目录至少需要包含：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- 一个或多个 `*.safetensors`

例如：

```text
Qwen3___5-9B-MLX-4bit/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
└── model-00002-of-00002.safetensors
```

---

## 4. 环境检查

先在目标环境中运行：

```bash
python scripts/preflight.py --strict
python scripts/verify_model.py
```

预期结果：

- `mlx: True (...)`
- `mlx_lm: True (...)`
- `ready: true`
- `status: ok`

如果这里失败，不要运行训练，先修环境。

---

## 5. 数据准备

首次使用先准备数据：

```bash
python prepare.py --augment
```

如果要启用 DeepSeek 蒸馏 API，请先配置 OpenAI 兼容参数。支持命令行参数或环境变量：

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_API_KEY="..."
python prepare.py --augment
```

也可以直接传参：

```bash
python prepare.py \
  --base-url "https://api.deepseek.com/v1" \
  --api-key "YOUR_API_KEY" \
  --augment
```

如果没有配置 `base_url` 和 `api_key`，`prepare.py` 会自动回退到内置模板生成模式，不会阻塞数据准备流程。

默认种子问题文件是：

- `data/seed_questions.txt`

需要调整默认 seed 问题时，直接编辑这个文件即可。

这一步会生成：

- `data/raw/distilled_latest.jsonl`
- `data/frozen/metadata.yaml`
- `data/frozen/valid_*.jsonl`
- `data/frozen/test_*.jsonl`
- `data/mlx_lm/train.jsonl`
- `data/mlx_lm/valid.jsonl`
- `data/mlx_lm/test.jsonl`

其中验证集和测试集是冻结数据，不应手工修改。

---

## 6. 训练命令

最小烟雾测试：

```bash
python train.py --preset smoke_real
```

基线训练：

```bash
python train.py --preset baseline
```

更长训练：

```bash
python train.py --preset deep_train
```

训练输出会落到：

- `outputs/adapters/<experiment_id>/`
- `outputs/results.tsv`

每次实验目录中会包含：

- `adapters.safetensors`
- `adapter_config.json`
- `metadata.json`

---

## 7. 评估命令

对某次训练结果做真实评估：

```bash
python scripts/evaluate_adapter.py \
  --adapter-path outputs/adapters/<experiment_id> \
  --model-path /absolute/path/to/model \
  --valid-file data/mlx_lm/valid.jsonl
```

---

## 8. 常见问题

### 8.1 `ModuleNotFoundError: No module named 'mlx'`

说明当前 Python 不是 MLX 训练环境。

处理方式：

```bash
pyenv activate mlx
python scripts/preflight.py --strict
```

### 8.2 `Model path does not exist`

说明 `config/model_config.yaml` 的路径不对，或者模型目录被移动了。

### 8.3 `adapter-path must be an experiment adapter directory`

评估脚本要求传实验目录，例如：

```bash
outputs/adapters/exp_20260330_143406_145647
```

不要直接传单个 `adapters.safetensors` 文件。

### 8.4 训练很慢

这是正常现象。9B 模型即使做很小的 smoke run，也可能需要较长时间完成加载和一次真实训练。

建议：

- 先跑 `smoke_real`
- 再跑 `baseline`
- 确认链路稳定后再做更长实验

---

## 9. 推荐执行顺序

```bash
pyenv activate mlx
python scripts/preflight.py --strict
python scripts/verify_model.py
python prepare.py --augment
python train.py --preset smoke_real
python scripts/evaluate_adapter.py \
  --adapter-path outputs/adapters/<experiment_id> \
  --model-path "$(python - <<'PY'
import yaml
with open('config/model_config.yaml', encoding='utf-8') as f:
    print(yaml.safe_load(f)['model']['path'])
PY
)"
```

---

## 10. 维护原则

后续如果迁移环境或模型路径，只改两类内容：

- 当前 shell / virtualenv / pyenv 激活方式
- `config/model_config.yaml` 的 `model.path`

不要再把环境切换逻辑塞回 `train.py`。
