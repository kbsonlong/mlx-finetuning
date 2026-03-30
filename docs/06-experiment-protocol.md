# MLX-LoRA 实验协议

> **创建日期**: 2026-03-30
> **用途**: 定义实验的评估方法、数据版本控制、结果记录标准

---

## 1. 数据版本控制

### 1.1 数据冻结原则

```
验证集和测试集一经生成即冻结，不允许被任何流程覆盖或修改。
只有训练集可以用于数据增强。
```

### 1.2 数据版本标识

每次实验必须记录以下数据版本信息：

```yaml
experiment_metadata:
  data:
    dataset_id: "ds_20260330_001"           # 数据集唯一ID
    data_hash: "a1b2c3d4e5f6"              # 数据内容哈希
    split_seed: 42                          # 数据分割随机种子
    split_ratio: [0.8, 0.1, 0.1]           # train/valid/test 比例

    prompt_template_version: "v1.0"         # prompt 模板版本
    teacher_model_version: "deepseek-v3"    # 蒸馏数据使用的教师模型
    distillation_date: "2026-03-30"         # 蒸馏日期

    train_samples: 9000                     # 训练集样本数
    valid_samples: 1000                     # 验证集样本数
    test_samples: 1000                      # 测试集样本数
```

### 1.3 数据目录结构

```
data/
├── frozen/                              # 冻结数据（不可修改）
│   ├── valid_20260330_001.jsonl         # 验证集（只读）
│   ├── test_20260330_001.jsonl          # 测试集（只读）
│   └── metadata.yaml                     # 数据元数据（只读）
├── mlx_lm/                              # MLX 格式数据
│   ├── train.jsonl                      # 训练集（可增强）
│   ├── valid.jsonl                      # 指向 frozen/ 的软链接
│   └── test.jsonl                       # 指向 frozen/ 的软链接
└── raw/                                 # 原始蒸馏数据
    └── distilled_20260330.jsonl
```

### 1.4 数据保护机制

```python
# scripts/data_guard.py
import os
import hashlib
from pathlib import Path
import yaml

FROZEN_DIR = Path("data/frozen")
METADATA_FILE = FROZEN_DIR / "metadata.yaml"

def load_data_metadata():
    """加载数据元数据"""
    if not METADATA_FILE.exists():
        raise FileNotFoundError("数据元数据不存在，请先生成数据")
    with open(METADATA_FILE) as f:
        return yaml.safe_load(f)

def compute_file_hash(filepath):
    """计算文件哈希"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_frozen_integrity():
    """验证冻结数据完整性"""
    metadata = load_data_metadata()

    for dataset_name, expected_hash in metadata["file_hashes"].items():
        filepath = FROZEN_DIR / dataset_name
        if not filepath.exists():
            raise FileNotFoundError(f"冻结数据缺失: {dataset_name}")

        actual_hash = compute_file_hash(filepath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"数据被篡改: {dataset_name}\n"
                f"  预期: {expected_hash}\n"
                f"  实际: {actual_hash}"
            )

    print("✅ 冻结数据完整性验证通过")
    return metadata

def protect_frozen_data():
    """设置冻结数据为只读"""
    metadata = load_data_metadata()

    for dataset_name in metadata["file_hashes"].keys():
        filepath = FROZEN_DIR / dataset_name
        os.chmod(filepath, 0o444)  # 只读权限

    print("✅ 冻结数据已设置为只读")
```

---

## 2. 随机种子控制

### 2.1 种子设置标准

```python
# 每次实验必须设置的随机种子
RANDOM_SEED = 42

import random
import numpy as np

# 在训练开始前设置
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# MLX 内部随机性
# MLX 暂不直接支持种子设置，通过固定数据顺序保证可重复性
```

### 2.2 实验可重复性

```yaml
# 每次实验记录
experiment:
  id: "exp_001"
  random_seed: 42
  cuda_seed: 42          # 如果使用 GPU
  python_hash_seed: 0    # Python 哈希种子

reproducibility:
  platform: "darwin"
  mlx_version: "0.18.0"
  python_version: "3.11"
```

---

## 3. 评估方法定义

### 3.1 监督微调评估协议

**核心原则**: 固定 tokenizer、固定 prompt template、按 token-level 负对数似然计算验证集 loss

```python
# scripts/evaluate_adapter.py
from mlx_lm import load
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
import json

def evaluate_adapter(
    adapter_path: str,
    valid_file: str,
    model_path: str,
    max_seq_length: int = 512,
    batch_size: int = 4
):
    """
    评估 LoRA 适配器性能

    Args:
        adapter_path: 适配器路径
        valid_file: 验证集文件（MLX JSONL 格式）
        model_path: 基础模型路径
        max_seq_length: 最大序列长度
        batch_size: 批次大小

    Returns:
        dict: 包含 val_loss, perplexity, tokens_evaluated 等指标
    """
    # 加载模型和适配器
    model, tokenizer = load(model_path)
    adapter = load_adapter(adapter_path)
    model = apply_lora(model, adapter)

    model.eval()

    # 加载验证数据
    with open(valid_file) as f:
        data = [json.loads(line) for line in f]

    total_loss = mx.array(0.0)
    total_tokens = 0
    num_batches = 0

    # 批次评估
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch = data[i:i + batch_size]

        # Tokenize
        texts = [item["text"] for item in batch]
        tokens = tokenizer(
            texts,
            return_tensors="mx",
            truncation=True,
            max_length=max_seq_length,
            padding=True
        )

        # 前向传播计算 loss
        with mx.stream(mx.gpu):  # 使用 GPU
            outputs = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                labels=tokens["input_ids"]  # 语言模型：预测下一个 token
            )

            # 计算 token-level 负对数似然
            loss = outputs.loss  # 平均 loss per token
            batch_tokens = tokens["input_ids"].shape[0] * tokens["input_ids"].shape[1]

            total_loss += loss * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

    # 聚合指标
    val_loss = (total_loss / total_tokens).item()
    perplexity = mx.exp(val_loss).item()

    return {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "tokens_evaluated": total_tokens,
        "num_batches": num_batches,
        "adapter_path": adapter_path
    }


def load_adapter(adapter_path: str):
    """加载适配器权重"""
    import mlx.nn as nn
    return nn.load_weights(adapter_path)


def apply_lora(model, adapter_weights):
    """应用 LoRA 适配器到模型"""
    # TODO: 实现 LoRA 应用逻辑
    return model
```

### 3.2 评估指标定义

| 指标 | 计算方法 | 用途 |
|------|----------|------|
| **val_loss** | token-level NLL 平均值 | 主要优化目标 |
| **perplexity** | exp(val_loss) | 可解释性指标 |
| **tokens_evaluated** | 验证集总 token 数 | 验证数据规模 |
| **eval_time_seconds** | 评估耗时 | 效率监控 |

### 3.3 Accuracy 的使用限制

```python
# ⚠️ val_accuracy 仅在结构化任务中使用
# 生成式任务不定义 accuracy

def evaluate_structured_task(predictions, labels):
    """
    仅用于分类、QA 等有标准答案的结构化任务
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(labels)
    return accuracy
```

---

## 4. 结果记录 Schema

### 4.1 实验结果格式

```python
# 实验结果记录到 outputs/results.jsonl
# 每行一个 JSON 对象

{
  "experiment_id": "exp_20260330_001",
  "timestamp": "2026-03-30T14:30:00Z",

  "data": {
    "dataset_id": "ds_20260330_001",
    "data_hash": "a1b2c3d4e5f6",
    "split_seed": 42,
    "valid_samples": 1000
  },

  "model": {
    "base_model": "Qwen3.5-9B-MLX-4bit",
    "adapter_path": "outputs/adapters/adapters.npz"
  },

  "config": {
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_layers": 16,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "iters": 100,
    "max_seq_length": 512,
    "optimizer": "adam",
    "random_seed": 42
  },

  "metrics": {
    "val_loss": 1.234567,
    "perplexity": 45.67,
    "tokens_evaluated": 512000,
    "eval_time_seconds": 120.5
  },

  "training": {
    "train_time_seconds": 540.0,
    "peak_memory_mb": 14336.0,
    "final_train_loss": 0.987654,
    "status": "completed"  # completed, failed, timeout
  },

  "git": {
    "commit_hash": "abc123def456",
    "branch": "autoresearch/20260330-001",
    "is_dirty": false
  },

  "description": "Initial baseline run"
}
```

### 4.2 结果查询工具

```python
# scripts/query_results.py
import json
from pathlib import Path
from typing import List, Dict, Optional

RESULTS_FILE = Path("outputs/results.jsonl")

def load_all_results() -> List[Dict]:
    """加载所有实验结果"""
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def get_best_results(n: int = 5, metric: str = "val_loss") -> List[Dict]:
    """获取最好的 N 个结果"""
    results = load_all_results()
    # 按 val_loss 升序排序（越小越好）
    sorted_results = sorted(
        results,
        key=lambda x: x["metrics"][metric]
    )
    return sorted_results[:n]

def get_results_by_config(config_filter: Dict) -> List[Dict]:
    """按配置筛选结果"""
    results = load_all_results()
    filtered = []
    for r in results:
        match = True
        for key, value in config_filter.items():
            if r["config"].get(key) != value:
                match = False
                break
        if match:
            filtered.append(r)
    return filtered

def print_summary(results: List[Dict]):
    """打印结果摘要"""
    print(f"{'Exp ID':<20} {'val_loss':<12} {'perplexity':<12} {'lr':<10} {'rank':<6}")
    print("-" * 70)
    for r in results[:10]:  # 显示前10个
        print(f"{r['experiment_id']:<20} "
              f"{r['metrics']['val_loss']:<12.6f} "
              f"{r['metrics']['perplexity']:<12.2f} "
              f"{r['config']['learning_rate']:<10.2e} "
              f"{r['config']['lora_rank']:<6}")
```

---

## 5. 实验数据集生成流程

### 5.1 首次数据生成

```bash
# 1. 生成蒸馏数据（使用 DeepSeek API）
python scripts/distill_data.py \
  --num-samples 10000 \
  --output data/raw/distilled_20260330.jsonl

# 2. 数据增强（仅用于训练集）
python scripts/augment_data.py \
  --input data/raw/distilled_20260330.jsonl \
  --augment-context \
  --augment-complexity

# 3. 分割数据（生成冻结的验证集和测试集）
python scripts/split_data.py \
  --input data/raw/distilled_20260330.jsonl \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --frozen-dir data/frozen \
  --output-dir data/mlx_lm

# 4. 转换为 MLX 格式
python scripts/convert_to_mlx.py \
  --train data/mlx_lm/train.jsonl \
  --valid data/mlx_lm/valid.jsonl \
  --test data/mlx_lm/test.jsonl
```

### 5.2 数据分割脚本

```python
# scripts/split_data.py
import json
import random
import yaml
import hashlib
from pathlib import Path

def split_data(
    input_file: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    frozen_dir: str = "data/frozen",
    output_dir: str = "data/mlx_lm"
):
    """分割数据集并冻结验证集和测试集"""
    random.seed(seed)

    # 加载数据
    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    # 打乱并分割
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    # 创建目录
    Path(frozen_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 生成数据集 ID
    dataset_id = f"ds_{Path(input_file).stem.split('_')[-1]}_{random.randint(1000, 9999)}"

    # 保存冻结数据（验证集和测试集）
    valid_file = Path(frozen_dir) / f"valid_{dataset_id}.jsonl"
    test_file = Path(frozen_dir) / f"test_{dataset_id}.jsonl"

    with open(valid_file, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存训练数据（可增强）
    train_file = Path(output_dir) / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 创建软链接（MLX 格式）
    (Path(output_dir) / "valid.jsonl").symlink_to(valid_file)
    (Path(output_dir) / "test.jsonl").symlink_to(test_file)

    # 计算文件哈希
    file_hashes = {
        valid_file.name: compute_file_hash(valid_file),
        test_file.name: compute_file_hash(test_file)
    }

    # 保存元数据
    metadata = {
        "dataset_id": dataset_id,
        "split_seed": seed,
        "split_ratio": [train_ratio, valid_ratio, test_ratio],
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "test_samples": len(test_data),
        "file_hashes": file_hashes,
        "generated_at": Path(input_file).stem
    }

    with open(Path(frozen_dir) / "metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)

    # 设置只读权限
    os.chmod(valid_file, 0o444)
    os.chmod(test_file, 0o444)

    print(f"✅ 数据集分割完成")
    print(f"   数据集 ID: {dataset_id}")
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(valid_data)} 样本 (已冻结)")
    print(f"   测试集: {len(test_data)} 样本 (已冻结)")

    return dataset_id
```

---

## 6. 实验流程检查清单

### 6.1 开始实验前

- [ ] 验证冻结数据完整性（`verify_frozen_integrity()`）
- [ ] 确认数据集 ID 已记录
- [ ] 设置随机种子
- [ ] 检查模型路径正确
- [ ] 确认输出目录存在

### 6.2 实验进行中

- [ ] 记录所有超参数配置
- [ ] 监控训练 loss 防止发散
- [ ] 记录内存使用峰值
- [ ] 保存适配器权重

### 6.3 实验完成后

- [ ] 在冻结验证集上评估
- [ ] 计算所有评估指标
- [ ] 记录结果到 results.jsonl
- [ ] 验证结果完整性
- [ ] 更新最佳配置（如果改进）

---

## 7. 命令行工具

```bash
# 验证数据完整性
python scripts/data_guard.py verify

# 查询最佳结果
python scripts/query_results.py --best 5

# 按配置筛选
python scripts/query_results.py --filter '{"lora_rank": 8}'

# 评估特定适配器
python scripts/evaluate_adapter.py \
  --adapter outputs/adapters/exp_001.npz \
  --valid data/mlx_lm/valid.jsonl \
  --model models/Qwen3.5-9B-MLX-4bit
```

---

**文档版本**: 1.0
**最后更新**: 2026-03-30
