# MLX-LoRA 自主微调系统 - 技术架构

> **创建日期**: 2026-03-30
> **版本**: 1.0

---

## 一、系统架构图

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Claude Code  │  │  CLI 命令行   │  │  Python API  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼─────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────────────┐
│                    控制层 (Control Plane)                            │
│  ┌───────────────────────────▼──────────────────────────────┐       │
│  │                    program.md                             │       │
│  │              (AI Agent 指令集)                             │       │
│  └───────────────────────────┬──────────────────────────────┘       │
│  ┌───────────────────────────▼──────────────────────────────┐       │
│  │                  实验循环控制器                            │       │
│  │  - 状态管理  - 决策逻辑  - Git 集成                        │       │
│  └─────┬─────────────┬─────────────┬─────────────┬──────────┘       │
└────────┼─────────────┼─────────────┼─────────────┼─────────────────┘
         │             │             │             │
┌────────▼─────────────▼─────────────▼─────────────▼─────────────────┐
│                      执行层 (Execution)                             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  train.py    │  │ prepare.py   │  │ scripts/evaluate_adapter.py │             │
│  │  (训练脚本)   │  │  (数据准备)   │  │  (评估脚本)   │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
└─────────┼──────────────────┼──────────────────┼────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────────┐
│                      MLX 框架层                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   MLX-LM     │  │    MLX       │  │   Metal GPU  │             │
│  │  (模型接口)   │  │  (张量计算)   │  │  (硬件加速)   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────────┐
│                      数据层 (Data)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ 训练数据      │  │ 验证数据      │  │   Git 仓库   │             │
│  │ data/mlx_lm  │  │ data/mlx_lm  │  │ .git/        │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流图

```
                    ┌─────────────────┐
                    │  DeepSeek API   │
                    │  (教师模型)      │
                    └────────┬────────┘
                             │ 蒸馏数据
                             ▼
                    ┌─────────────────┐
                    │  数据增强模块     │
                    │  - 上下文增强     │
                    │  - 复杂度增强     │
                    │  - 多轮对话      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  MLX 格式转换    │
                    │  - Train/Valid   │
                    │  - JSONL 格式    │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │ 训练集    │ │ 验证集    │ │ 测试集    │
         └──────────┘ └──────────┘ └──────────┘
                │            │
                └────────────┼────────────┐
                             │            │
                             ▼            ▼
                    ┌─────────────────┐  │
                    │   MLX-LoRA      │  │
                    │   训练循环       │  │
                    │   - 前向传播      │  │
                    │   - 反向传播      │  │
                    │   - 参数更新      │  │
                    └────────┬────────┘  │
                             │            │
                             ▼            │
                    ┌─────────────────┐  │
                    │   LoRA 适配器    │  │
                    │   outputs/       │  │
                    └────────┬────────┘  │
                             │            │
                             └────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │   评估模块       │
                             │   - Loss 计算    │
                             │   - Accuracy     │
                             │   - Perplexity   │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │   决策模块       │
                             │   - 保留/丢弃    │
                             │   - Git 操作     │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │ outputs/        │
                             │ results.jsonl   │
                             │ (实验历史)       │
                             └─────────────────┘
```

---

## 二、核心模块设计

### 2.1 prepare.py - 数据准备模块

```python
"""
prepare.py - 一次性数据准备脚本

功能:
1. 从 DeepSeek API 蒸馏数据
2. 数据增强
3. 转换为 MLX 格式
4. 分割 Train/Valid/Test
"""

import argparse
from pathlib import Path
from scripts.deepseek_distill import DeepSeekDistiller
from scripts.augment_distilled import DistilledDataAugmentor
from scripts.prepare_mlx_lm_dataset import main as prepare_mlx

def parse_args():
    parser = argparse.ArgumentParser()
    # DeepSeek 配置
    parser.add_argument("--api-key", help="DeepSeek API Key")
    parser.add_argument("--questions", default="data/seed_questions.txt")
    parser.add_argument("--num-samples", type=int, default=500)

    # 数据增强
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-strategies", nargs="+",
                        default=["context", "complexity"])

    # 输出
    parser.add_argument("--output-dir", default="data/mlx_lm")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. DeepSeek 蒸馏
    print("🔬 开始 DeepSeek 蒸馏...")
    distiller = DeepSeekDistiller(api_key=args.api_key)
    distilled_file = "data/distilled.jsonl"
    distiller.batch_distill(
        questions=load_questions(args.questions),
        output_path=distilled_file,
        num_samples=args.num_samples
    )

    # 2. 数据增强
    if args.augment:
        print("🔄 开始数据增强...")
        augmentor = DistilledDataAugmentor(distilled_file)
        augmented_data = augmentor.augment(args.augment_strategies)
        augmented_file = "data/augmented.jsonl"
        augmentor.save(augmented_file, augmented_data)
    else:
        augmented_file = distilled_file

    # 3. MLX 格式转换
    print("📦 转换为 MLX 格式...")
    prepare_mlx(
        input=augmented_file,
        output_dir=args.output_dir
    )

    print(f"✅ 数据准备完成: {args.output_dir}")

if __name__ == "__main__":
    main()
```

### 2.2 train.py - 训练脚本 (AI 修改目标)

```python
"""
train.py - MLX-LoRA 训练脚本

这是 AI Agent 修改的主要文件。
只有顶部的配置区可以被修改。
"""

# ============================================================================
# 可配置区 - AI 可以修改这部分
# ============================================================================

# LoRA 配置
LORA_RANK = 8           # LoRA 秩 (可调: 4, 8, 16, 32)
LORA_ALPHA = 16         # LoRA alpha (可调: 8, 16, 32, 64)
LORA_DROPOUT = 0.05     # Dropout 率 (可调: 0.0, 0.05, 0.1)

# 训练超参数
LEARNING_RATE = 5e-5    # 学习率 (可调: 1e-5, 3e-5, 5e-5, 1e-4)
BATCH_SIZE = 4          # 批大小 (可调: 2, 4, 8)
ITERS = 100             # 迭代次数 (可调: 50, 100, 200)
MAX_SEQ_LENGTH = 512    # 序列长度 (可调: 256, 512, 1024)

# 优化器配置
OPTIMIZER = "adam"      # 优化器 (可调: adam, adamw)
GRAD_CHECKPOINT = True  # 梯度检查点 (可调: True, False)

# 学习率调度
LR_SCHEDULE = "constant"  # 调度策略 (可调: constant, cosine, linear)

# ============================================================================
# 固定区 - 以下代码不应被修改
# ============================================================================

import os
import time
import json
from pathlib import Path
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.utils import load_adapter, save_adapter
import tqdm

# 加载配置
import yaml

def load_model_config():
    """加载模型配置 (本地模型路径)"""
    model_config_path = Path("config/model_config.yaml")

    if model_config_path.exists():
        with open(model_config_path) as f:
            return yaml.safe_load(f)

    # 默认配置
    return {
        "model": {
            "path": "models/Qwen3.5-9B-MLX-4bit",
            "name": "Qwen3.5-9B-MLX-4bit"
        }
    }

def load_config():
    """加载基础配置"""
    config_path = Path("config/base_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 合并模型配置 (使用本地模型路径)
    model_config = load_model_config()
    config["model_path"] = model_config["model"]["path"]

    return config

def load_data(split):
    """加载 MLX 格式数据"""
    data_path = Path(f"data/mlx_lm/{split}.jsonl")
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_model(config):
    """创建模型和 LoRA 适配器"""
    # 使用本地模型路径
    model, tokenizer = load(config["model_path"])

    # 配置 LoRA
    lora_config = {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "layers": config.get("num_layers", 16),
    }

    # 应用 LoRA
    from mlx_lm.utils import apply_lora
    model = apply_lora(model, lora_config)

    return model, tokenizer

def train_step(model, batch, optimizer, loss_fn):
    """单步训练"""
    def loss_fn_wrapper(model, batch):
        return loss_fn(model, batch)

    loss, grads = loss_fn_wrapper(model, batch)
    optimizer.update(model, grads)
    return loss

def evaluate(model, tokenizer, valid_data):
    """验证集评估"""
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for item in valid_data:
        prompt = item["prompt"]
        completion = item["completion"]

        # 生成预测
        input_ids = tokenizer.encode(prompt)
        output = generate(model, tokenizer, prompt=prompt, max_tokens=256)

        # 计算 loss (简化)
        # 实际实现需要更精确的计算

    # 返回评估指标
    val_loss = total_loss / len(valid_data)
    val_accuracy = total_correct / total_tokens
    perplexity = mx.exp(val_loss)

    return {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "perplexity": perplexity,
    }

def main():
    # 加载配置
    config = load_config()

    # 加载数据
    print("📚 加载数据...")
    train_data = load_data("train")
    valid_data = load_data("valid")

    # 创建模型
    print("🤖 创建模型...")
    model, tokenizer = create_model(config)

    # 配置优化器
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    else:
        optimizer = optim.AdamW(learning_rate=LEARNING_RATE)

    # 训练循环
    print("🚀 开始训练...")
    start_time = time.time()

    losses = []
    for step in tqdm(range(ITERS)):
        # 采样 batch
        batch_indices = mx.random.randint(0, len(train_data), [BATCH_SIZE])
        batch = [train_data[i] for i in batch_indices]

        # 训练步骤
        loss = train_step(model, batch, optimizer, lambda m, b: m(b))
        losses.append(loss)

        # 定期评估
        if (step + 1) % 10 == 0:
            avg_loss = mx.stack(losses).mean()
            print(f"Step {step+1}/{ITERS}, Loss: {avg_loss:.4f}")
            losses = []

    # 最终评估
    print("📊 最终评估...")
    metrics = evaluate(model, tokenizer, valid_data)

    # 保存适配器
    output_dir = Path("outputs/adapters")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_adapter(model, str(adapter_path))

    # 输出结果
    train_time = time.time() - start_time
    print("---")
    print(f"val_loss:        {metrics['val_loss']:.6f}")
    print(f"val_accuracy:    {metrics['val_accuracy']:.4f}")
    print(f"perplexity:      {metrics['perplexity']:.2f}")
    print(f"train_time:      {train_time:.1f}")
    print(f"lora_rank:       {LORA_RANK}")
    print(f"lora_alpha:      {LORA_ALPHA}")
    print(f"learning_rate:   {LEARNING_RATE}")
    print(f"batch_size:      {BATCH_SIZE}")
    print(f"iters:           {ITERS}")

if __name__ == "__main__":
    main()
```

### 2.3 program.md - AI Agent 指令集

```markdown
# MLX-LoRA 自主微调研究系统

## 系统概述

这是一个自主 LoRA 微调参数研究系统。你的目标是通过反复实验，
找到最佳的 LoRA 配置和训练超参数，以最小化验证集损失。

## 实验约束

### 时间预算
- 每次实验限制: 10 分钟
- 超时处理: 杀死进程并标记为失败

### 可修改文件
- **唯一可修改**: `train.py` 顶部配置区
- **只读文件**: `prepare.py`, `config/`, `scripts/`

### 可调参数

| 参数 | 可选值 | 说明 |
|------|--------|------|
| LORA_RANK | 4, 8, 16, 32 | LoRA 秩 |
| LORA_ALPHA | 8, 16, 32, 64 | LoRA alpha |
| LORA_DROPOUT | 0.0, 0.05, 0.1 | Dropout |
| LEARNING_RATE | 1e-5, 3e-5, 5e-5, 1e-4 | 学习率 |
| BATCH_SIZE | 2, 4, 8 | 批大小 |
| ITERS | 50, 100, 200 | 迭代次数 |
| MAX_SEQ_LENGTH | 256, 512, 1024 | 序列长度 |
| OPTIMIZER | adam, adamw | 优化器 |
| GRAD_CHECKPOINT | True, False | 梯度检查点 |
| LR_SCHEDULE | constant, cosine, linear | 学习率调度 |

## 实验循环

### 1. 状态分析
```bash
# 查看当前 Git 状态
git status
git log --oneline -5

# 查看历史结果
cat outputs/results.jsonl | jq -s '. | sort_by(.metrics.val_loss) | .[0:5]'
```

### 2. 生成假设
基于历史结果，决定下一步尝试:
- 如果 loss 高 → 尝试增加容量 (rank/alpha)
- 如果过拟合 → 增加 dropout
- 如果欠拟合 → 增加 iters/学习率
- 如果不稳定 → 降低学习率

### 3. 修改代码
编辑 `train.py` 顶部配置区:
```python
LORA_RANK = 16  # 从 8 增加
LEARNING_RATE = 3e-5  # 从 5e-5 降低
```

### 4. 提交更改
```bash
git add train.py
git commit -m "exp: increase lora_rank 8->16, lower lr 5e-5->3e-5"
```

### 5. 运行实验
```bash
python train.py > run.log 2>&1
```

### 6. 提取结果
```bash
grep "^val_loss:" run.log
grep "^perplexity:" run.log
```

### 7. 决策
```python
# 伪代码
if new_val_loss < best_val_loss:
    # 保留: 已经 commit，无需操作
    log_result("keep", "改进: val_loss X -> Y")
else:
    # 丢弃
    git reset --hard HEAD~1
    log_result("discard", "退步: val_loss X -> Y")
```

## 决策规则

### 主指标
- **val_loss**: 越低越好 (主要优化目标)

### 辅助指标
- **perplexity**: 应与 val_loss 相关
- **val_accuracy**: 越高越好
- **train_time**: 效率参考

### 约束条件
- **内存**: < 16GB
- **时间**: < 600 秒

### 简化奖励
当 val_loss 相近 (<1% 差异) 时，优先选择:
1. 更简单的配置 (更低的 rank/alpha)
2. 更少的参数
3. 更快的训练

## 常见策略

### 探索阶段 (初期)
- 尝试不同的 rank/alpha 组合
- 测试不同学习率
- 建立性能基准

### 利用阶段 (中期)
- 在最佳配置附近微调
- 细化学习率
- 测试正则化

### 优化阶段 (后期)
- 微调最佳参数
- 测试学习率调度
- 验证稳定性

## 紧急处理

### 崩溃处理
1. 查看 `run.log` 错误信息
2. 如果是 OOM: 降低 batch_size 或 rank
3. 如果是 NaN: 降低学习率
4. 重试 3 次，仍失败则跳过

### 无进展处理
如果连续 10 次实验无改进:
1. 尝试更大幅度的参数变化
2. 尝试完全不同的配置
3. 检查数据是否有问题

## 持续运行

**重要**: 一旦实验开始，持续运行直到:
- 用户手动中断 (Ctrl+C)
- 系统关机
- 遇到无法恢复的错误

不要:
- 停下来询问是否继续
- 等待用户确认
- 自己决定"足够好了"

## 记录格式

outputs/results.jsonl 格式:
```json
{"experiment_id": "exp_001", "config": {"lora_rank": 8, ...}, "metrics": {"val_loss": 1.234567, ...}, "status": "completed", "branch": "exp/exp_001"}
{"experiment_id": "exp_002", "config": {"lora_rank": 16, ...}, "metrics": {"val_loss": 1.230000, ...}, "status": "discarded", "branch": "exp/exp_002"}
```

---

## 开始使用

1. 确认数据已准备: `ls data/mlx_lm/`
2. 创建输出目录: `mkdir -p outputs`
3. 运行 baseline: `python train.py > outputs/run.log 2>&1`
4. 开始自主实验循环
```

---

## 三、关键技术点

### 3.1 MLX LoRA 实现

```python
# MLX LoRA 核心实现
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 原始权重 (冻结)
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA 低秩分解
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # 初始化
        nn.normal(self.lora_A.weight, 0.0, 0.01)
        nn.zeros(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def __call__(self, x):
        # 原始变换
        result = self.linear(x)

        # LoRA 增量
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        result = result + lora_out * self.scaling

        return result
```

### 3.2 评估指标计算

```python
def compute_metrics(model, tokenizer, data, batch_size=4):
    """计算评估指标"""
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        # 准备输入
        prompts = [item["prompt"] for item in batch]
        targets = [item["completion"] for item in batch]

        # 前向传播
        with mx.stream(mx.gpu):
            inputs = [tokenizer.encode(p) for p in prompts]
            outputs = model(inputs)

            # 计算损失
            loss = cross_entropy_loss(outputs, targets)
            total_loss += loss

            # 计算准确率
            predictions = tokenizer.decode(outputs.argmax(-1))
            total_correct += sum(p == t for p, t in zip(predictions, targets))
            total_tokens += sum(len(t) for t in targets)

    # 汇总指标
    metrics = {
        "val_loss": total_loss / len(data),
        "val_accuracy": total_correct / total_tokens,
        "perplexity": mx.exp(total_loss / len(data)),
    }

    return metrics
```

### 3.3 Git 集成

```python
import subprocess
from typing import Optional

def get_current_commit() -> str:
    """获取当前 commit hash"""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def commit_experiment(description: str) -> str:
    """提交实验更改"""
    # 添加 train.py
    subprocess.run(["git", "add", "train.py"], check=True)

    # 提交
    subprocess.run(
        ["git", "commit", "-m", f"exp: {description}"],
        check=True
    )

    return get_current_commit()

def reset_experiment():
    """回退实验更改"""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        check=True
    )

def create_experiment_branch(tag: str):
    """创建实验分支"""
    branch_name = f"autoresearch/{tag}"
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        check=True
    )
    return branch_name
```

---

## 四、性能优化

### 4.1 内存优化

```python
# 梯度检查点
GRAD_CHECKPOINT = True  # 节省内存，略微增加计算时间

# 混合精度训练
mx.default_device(mx.gpu)
model = model.to(mx.float16)  # 使用 FP16

# 梯度累积
EFFECTIVE_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
```

### 4.2 训练加速

```python
# 数据预取
class DataLoader:
    def __init__(self, data, batch_size, prefetch=2):
        self.data = data
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.buffer = []

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size]
            yield batch

# 编译优化
@mx.compile
def training_step(model, batch):
    loss = model(batch)
    return loss
```

---

## 五、监控与调试

### 5.1 实时监控

```python
# 训练进度条
from tqdm import tqdm

for step in tqdm(range(ITERS), desc="Training"):
    loss = train_step(model, batch)
    tqdm.set_postfix({"loss": f"{loss:.4f}"})

# TensorBoard 日志
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/")
writer.add_scalar("Loss/train", loss, step)
writer.add_scalar("Loss/val", val_loss, step)
```

### 5.2 错误处理

```python
import sys
import traceback

def safe_train():
    try:
        main()
    except mx.MXError as e:
        if "out of memory" in str(e):
            print("❌ OOM: 降低 batch_size 或 rank")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    safe_train()
```

---

**文档版本**: 1.0
**最后更新**: 2026-03-30
