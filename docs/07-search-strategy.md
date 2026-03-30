# MLX-LoRA 参数搜索策略

> **创建日期**: 2026-03-30
> **用途**: 定义高效的参数搜索方法和时间预算分配

---

## 1. 搜索空间定义

### 1.1 完整参数空间

```yaml
search_space:
  lora_rank: [4, 8, 16, 32]           # 4 个选项
  lora_alpha: [8, 16, 32, 64]         # 4 个选项
  lora_dropout: [0.0, 0.05, 0.1]      # 3 个选项
  learning_rate: [1e-5, 3e-5, 5e-5, 1e-4]  # 4 个选项
  batch_size: [2, 4, 8]               # 3 个选项
  iters: [50, 100, 200]               # 3 个选项
  max_seq_length: [256, 512, 1024]    # 3 个选项
  optimizer: ["adam", "adamw"]        # 2 个选项
  grad_checkpoint: [true, false]      # 2 个选项
  warmup_ratio: [0.0, 0.1, 0.2]       # 3 个选项

# 理论组合数: 4×4×3×4×3×3×3×2×2×3 = 62,208
```

### 1.2 实际约束

- 单次实验时间预算: 10-15 分钟
- 一晚可用时间: ~8 小时
- 目标: 一晚完成 100+ 次实验

**问题**: 完整网格搜索不现实

**解决方案**: 两阶段搜索策略

---

## 2. 两阶段搜索策略

### 2.1 策略概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     两阶段搜索流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段 1: 快速筛选 (Stage 1: Quick Screen)                        │
│  ├─ 时间预算: 5 分钟/实验                                        │
│  ├─ 样本数: 100-200 次随机采样                                  │
│  ├─ 总耗时: 8-16 小时                                           │
│  └─ 输出: Top 10-20 个候选配置                                  │
│                                                                 │
│                              ↓                                  │
│                                                                 │
│  阶段 2: 精细验证 (Stage 2: Fine Validation)                     │
│  ├─ 时间预算: 15-20 分钟/实验                                    │
│  ├─ 样本数: Top 10-20 配置 × 2-3 次重复                         │
│  ├─ 总耗时: 5-10 小时                                            │
│  └─ 输出: 最优配置 + 统计显著性                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 阶段 1: 快速筛选

**目标**: 快速识别有前景的配置区域

#### 配置

```yaml
stage1_quick_screen:
  # 缩短的训练配置
  time_budget_minutes: 5          # 每次实验 5 分钟
  iters: 50                       # 减少迭代次数
  max_seq_length: 256             # 缩短序列长度
  valid_samples: 500              # 采样验证集

  # 采样策略
  sampling_method: "random"       # 随机采样
  n_samples: 150                  # 采样 150 个配置

  # 固定参数 (基于基准测试)
  fixed:
    optimizer: "adam"
    grad_checkpoint: true

  # 随机参数
  random:
    lora_rank: [4, 8, 16, 32]
    lora_alpha: [8, 16, 32, 64]
    lora_dropout: [0.0, 0.05, 0.1]
    learning_rate: [1e-5, 3e-5, 5e-5, 1e-4]
    batch_size: [4, 8]
    warmup_ratio: [0.0, 0.1, 0.2]

  # 预期总耗时
  estimated_time_hours: 12.5      # 150 × 5 分钟 = 12.5 小时
```

#### 输出

```json
// outputs/stage1_results.json
{
  "stage": "quick_screen",
  "n_experiments": 150,
  "top_candidates": [
    {
      "rank": 1,
      "config": {"lora_rank": 16, "lora_alpha": 32, ...},
      "val_loss": 1.234,
      "score": 0.95
    },
    // ... Top 20
  ]
}
```

### 2.3 阶段 2: 精细验证

**目标**: 对 Top 配置进行充分验证

#### 配置

```yaml
stage2_fine_validation:
  # 完整训练配置
  time_budget_minutes: 15         # 每次实验 15 分钟
  iters: 100                      # 完整迭代次数
  max_seq_length: 512             # 完整序列长度
  valid_samples: -1               # 全量验证集

  # 验证策略
  candidates_from_stage: 20       # 从阶段 1 取 Top 20
  repeats_per_config: 3           # 每个配置重复 3 次

  # 随机种子控制
  seeds: [42, 123, 456]           # 固定种子

  # 预期总耗时
  estimated_time_hours: 15        # 20 × 3 × 15 分钟 = 15 小时
```

#### 输出

```json
// outputs/stage2_results.json
{
  "stage": "fine_validation",
  "n_configs": 20,
  "repeats_per_config": 3,
  "final_ranking": [
    {
      "rank": 1,
      "config": {...},
      "metrics": {
        "val_loss_mean": 1.234,
        "val_loss_std": 0.012,
        "val_loss_min": 1.218,
        "val_loss_max": 1.250
      },
      "stability": "high"  // high, medium, low
    }
  ]
}
```

---

## 3. 实现细节

### 3.1 随机采样实现

```python
# scripts/search.py
import random
import yaml
from pathlib import Path

def load_search_space():
    """加载参数搜索空间"""
    config_path = Path("config/search_space.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def random_sample_config(search_space, fixed_params=None):
    """从搜索空间随机采样一个配置"""
    config = {}

    # 应用固定参数
    if fixed_params:
        config.update(fixed_params)

    # 随机采样
    for param, values in search_space.get("random", {}).items():
        config[param] = random.choice(values)

    return config

def generate_stage1_configs(n_samples=150):
    """生成阶段 1 的实验配置"""
    search_space = load_search_space()
    stage1_config = search_space["stage1_quick_screen"]

    configs = []
    for i in range(n_samples):
        config = {
            "experiment_id": f"stage1_exp_{i:04d}",
            "stage": "quick_screen",
            **random_sample_config(
                stage1_config,
                stage1_config.get("fixed", {})
            )
        }
        configs.append(config)

    return configs

def generate_stage2_configs(stage1_results, top_n=20, repeats=3):
    """生成阶段 2 的实验配置"""
    search_space = load_search_space()
    stage2_config = search_space["stage2_fine_validation"]

    # 从阶段 1 结果选择 Top N
    top_configs = sorted(
        stage1_results,
        key=lambda x: x["val_loss"]
    )[:top_n]

    configs = []
    for rank, base_config in enumerate(top_configs):
        for repeat in range(repeats):
            config = {
                "experiment_id": f"stage2_rank{rank}_rep{repeat}",
                "stage": "fine_validation",
                "base_config": base_config["config"],
                "seed": stage2_config["seeds"][repeat],
                **base_config["config"]
            }
            configs.append(config)

    return configs
```

### 3.2 时间预算实现

```python
# scripts/budget.py
import time
from pathlib import Path
import signal

class TimeBudgetExceeded(Exception):
    pass

def time_budget_handler(signum, frame):
    raise TimeBudgetExceeded("Time budget exceeded")

def run_with_budget(func, budget_minutes=5):
    """在时间预算内运行函数"""
    # 设置超时
    signal.signal(signal.SIGALRM, time_budget_handler)
    signal.alarm(budget_minutes * 60)

    try:
        result = func()
        signal.alarm(0)  # 取消超时
        return result
    except TimeBudgetExceeded:
        return {"status": "timeout", "elapsed": budget_minutes * 60}
    except Exception as e:
        signal.alarm(0)
        return {"status": "failed", "error": str(e)}

# 使用示例
def train_experiment(config):
    """训练单次实验"""
    start_time = time.time()

    # ... 训练逻辑 ...

    elapsed = time.time() - start_time
    return {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "val_loss": 1.234
    }
```

### 3.3 结果聚合与分析

```python
# scripts/analyze.py
import json
import numpy as np
from pathlib import Path

def load_stage1_results():
    """加载阶段 1 结果"""
    results_file = Path("outputs/stage1_results.jsonl")
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def select_top_candidates(results, top_n=20):
    """选择 Top N 候选配置"""
    sorted_results = sorted(results, key=lambda x: x["metrics"]["val_loss"])
    return sorted_results[:top_n]

def aggregate_stage2_results(config_id, stage2_results):
    """聚合阶段 2 的重复实验结果"""
    config_results = [
        r for r in stage2_results
        if r["base_config_id"] == config_id
    ]

    val_losses = [r["metrics"]["val_loss"] for r in config_results]

    return {
        "config_id": config_id,
        "n_repeats": len(val_losses),
        "val_loss_mean": np.mean(val_losses),
        "val_loss_std": np.std(val_losses),
        "val_loss_min": np.min(val_losses),
        "val_loss_max": np.max(val_losses),
        "stability": classify_stability(np.std(val_losses))
    }

def classify_stability(std):
    """根据标准差分类稳定性"""
    if std < 0.01:
        return "high"
    elif std < 0.03:
        return "medium"
    else:
        return "low"
```

---

## 4. 搜索流程控制

### 4.1 自动化流程脚本

```bash
#!/bin/bash
# scripts/run_two_stage_search.sh

set -e

echo "=== 两阶段参数搜索 ==="

# 阶段 1: 快速筛选
echo "阶段 1: 快速筛选 (150 个配置 × 5 分钟)"
python scripts/generate_configs.py --stage 1 --n-samples 150
python scripts/run_experiments.py --config outputs/stage1_configs.jsonl --budget 5
python scripts/analyze_stage1.py

# 阶段 2: 精细验证
echo "阶段 2: 精细验证 (Top 20 配置 × 3 重复 × 15 分钟)"
python scripts/generate_configs.py --stage 2 --top-n 20 --repeats 3
python scripts/run_experiments.py --config outputs/stage2_configs.jsonl --budget 15
python scripts/analyze_stage2.py

# 最终报告
echo "生成最终报告"
python scripts/generate_final_report.py

echo "完成！"
```

### 4.2 进度监控

```python
# scripts/monitor.py
import json
import time
from pathlib import Path

def print_search_progress(results_file):
    """打印搜索进度"""
    completed = []
    with open(results_file) as f:
        for line in f:
            result = json.loads(line)
            if result["status"] in ["completed", "failed", "timeout"]:
                completed.append(result)

    n_total = 150  # 阶段 1
    n_done = len(completed)
    progress = n_done / n_total * 100

    # 统计
    n_success = sum(1 for r in completed if r["status"] == "completed")
    n_failed = sum(1 for r in completed if r["status"] == "failed")
    n_timeout = sum(1 for r in completed if r["status"] == "timeout")

    best_val_loss = min(
        (r["metrics"]["val_loss"] for r in completed if r["status"] == "completed"),
        default=float("inf")
    )

    print(f"进度: {n_done}/{n_total} ({progress:.1f}%)")
    print(f"  成功: {n_success}, 失败: {n_failed}, 超时: {n_timeout}")
    print(f"  当前最佳: {best_val_loss:.6f}")

# 持续监控
while True:
    print_search_progress("outputs/stage1_results.jsonl")
    time.sleep(60)  # 每分钟更新
```

---

## 5. 调优策略

### 5.1 早停条件

```python
# 如果前 50 个实验中，最好的 val_loss 没有明显改善
def should_stop_early(results, patience=50, threshold=0.01):
    """判断是否应该提前停止"""
    if len(results) < patience:
        return False

    recent = results[-patience:]
    best_recent = min(r["metrics"]["val_loss"] for r in recent)
    best_overall = min(r["metrics"]["val_loss"] for r in results)

    return (best_overall - best_recent) < threshold
```

### 5.2 自适应采样

```python
# 根据已有结果动态调整采样策略
def adaptive_sample_space(results):
    """根据结果调整采样空间"""
    # 如果低 learning_rate 表现好，缩小 lr 范围
    best_configs = sorted(results, key=lambda x: x["val_loss"])[:10]

    best_lrs = [c["config"]["learning_rate"] for c in best_configs]
    if all(lr < 3e-5 for lr in best_lrs):
        # 缩小到低学习率范围
        return {"learning_rate": [5e-6, 1e-5, 2e-5]}

    return None
```

---

## 6. 验收标准

### 阶段 1 完成标准

- [ ] 完成 100+ 次快速实验
- [ ] 识别出 Top 10-20 个候选配置
- [ ] 结果记录到 `outputs/stage1_results.jsonl`

### 阶段 2 完成标准

- [ ] Top 20 配置各完成 2-3 次重复实验
- [ ] 计算每个配置的均值和标准差
- [ ] 选择统计上最优的配置

### 最终输出

- [ ] 最优配置文件: `config/best_config.yaml`
- [ ] 完整实验报告: `outputs/final_report.md`
- [ ] 可视化图表: `outputs/plots/`

---

## 7. 时间预算总结

| 阶段 | 实验 | 时间/实验 | 总耗时 |
|------|------|-----------|--------|
| 阶段 1 | 150 | 5 分钟 | ~12.5 小时 |
| 阶段 2 | 20 × 3 | 15 分钟 | ~15 小时 |
| **总计** | **210** | - | **~27.5 小时** |

**建议**: 可以分两晚完成，每晚约 14 小时

---

**文档版本**: 1.0
**最后更新**: 2026-03-30
