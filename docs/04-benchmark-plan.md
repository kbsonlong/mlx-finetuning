# MLX-LoRA 自主微调系统 - 基准测试方案

> **创建日期**: 2026-03-30
> **目的**: 建立性能基线，确定合理的时间预算和优化目标

---

## 一、基准测试目标

### 1.1 测试目标

```
┌─────────────────────────────────────────────────────────────┐
│                     基准测试目的                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 建立基线性能        → 知道"起点"在哪里                  │
│  2. 测量单次实验时长      → 设置合理的时间预算              │
│  3. 收集基础指标数据      → 定义优化目标                    │
│  4. 验证系统稳定性        → 确保长时间运行不崩溃             │
│  5. 识别性能瓶颈        → 找到优化方向                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心问题

基准测试需要回答以下问题：

| 问题 | 为什么重要 | 如何测量 |
|------|-----------|----------|
| 单次训练需要多久？ | 设置实验时间预算 | 运行完整训练并计时 |
| 不同参数如何影响训练时间？ | 预估参数搜索空间 | 多组配置对比 |
| 基线 val_loss 是多少？ | 确定改进目标 | 默认配置训练 |
| 内存使用情况如何？ | 设置资源约束 | 监控峰值内存 |
| 系统是否稳定？ | 确保长时间运行 | 连续多次实验 |

---

## 二、环境准备

### 2.1 本地模型配置

**重要**: 为避免下载时间过长，本项目使用**本地已下载的模型**。

#### 模型目录结构

```
models/
└── Qwen3.5-9B-MLX-4bit/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

#### 配置本地模型路径

创建 `config/model_config.yaml`:

```yaml
# config/model_config.yaml
model:
  # 本地模型路径 (绝对路径或相对路径)
  path: "models/Qwen3.5-9B-MLX-4bit"

  # 模型信息
  name: "Qwen3.5-9B-MLX-4bit"
  size_gb: 20
  quantization: "4bit"

  # 验证文件存在
  required_files:
    - "config.json"
    - "model.safetensors"
    - "tokenizer.json"
```

#### 模型验证脚本

```python
# scripts/verify_model.py
import os
import yaml
from pathlib import Path

def verify_model_config():
    """验证本地模型配置"""

    # 加载配置
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)

    model_path = Path(config["model"]["path"])

    print(f"🔍 验证模型: {model_path}")

    # 检查目录存在
    if not model_path.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        return False

    # 检查必需文件
    for file in config["model"]["required_files"]:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file} 不存在")
            return False

    # 计算总大小
    total_size = sum(
        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
    ) / (1024**3)

    print(f"\n✅ 模型验证成功!")
    print(f"   总大小: {total_size:.2f} GB")
    print(f"   路径: {model_path.absolute()}")

    return True

if __name__ == "__main__":
    verify_model_config()
```

#### 使用本地模型

```python
# 加载模型时使用本地路径
from mlx_lm import load
import yaml

# 读取配置
with open("config/model_config.yaml") as f:
    config = yaml.safe_load(f)

# 加载本地模型
model, tokenizer = load(config["model"]["path"])
```

### 2.2 环境检查清单

在开始基准测试前，请确认：

- [ ] 本地模型已下载并验证
- [ ] 运行 `python scripts/verify_model.py` 确认模型完整
- [ ] MLX 已正确安装 (`python -c "import mlx; print(mlx.__version__)"`)
- [ ] 训练数据已准备 (`ls data/mlx_lm/`)
- [ ] 可用内存 > 32GB (`vm_stat | head -5`)
- [ ] 磁盘空间 > 50GB (`df -h`)

---

## 三、基准配置设计

### 3.1 默认基线配置

```yaml
# config/baseline_config.yaml
baseline:
  # 使用本地模型 (从 model_config.yaml 读取)
  model_path: "models/Qwen3.5-9B-MLX-4bit"
  data: "data/mlx_lm"

  # LoRA 配置 (保守起点)
  lora:
    rank: 8          # 中等规模
    alpha: 16        # 2x rank
    dropout: 0.05    # 轻微正则化
    layers: 16       # 前 16 层

  # 训练配置 (平衡点)
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

### 3.2 测试配置矩阵

| 配置名 | LORA_RANK | BATCH_SIZE | ITERS | MAX_SEQ_LEN | 目的 |
|--------|-----------|------------|-------|-------------|------|
| baseline | 8 | 4 | 100 | 512 | 基线参考 |
| fast_test | 4 | 8 | 50 | 256 | 快速验证 |
| large_rank | 16 | 4 | 100 | 512 | 大秩测试 |
| long_seq | 8 | 4 | 100 | 1024 | 长序列测试 |
| deep_train | 8 | 4 | 200 | 512 | 深度训练 |
| small_batch | 8 | 2 | 100 | 512 | 小批量测试 |

### 3.3 配置加载工具

```python
# utils/config.py
import yaml
from pathlib import Path
from typing import Dict

def load_model_config() -> Dict:
    """加载模型配置"""
    config_path = Path("config/model_config.yaml")

    if not config_path.exists():
        # 默认配置
        return {
            "model": {
                "path": "models/Qwen3.5-9B-MLX-4bit",
                "name": "Qwen3.5-9B-MLX-4bit"
            }
        }

    with open(config_path) as f:
        return yaml.safe_load(f)

def load_baseline_config() -> Dict:
    """加载基线配置"""
    config_path = Path("config/baseline_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 确保模型路径正确
    model_config = load_model_config()
    config["baseline"]["model_path"] = model_config["model"]["path"]

    return config["baseline"]

# 使用示例
if __name__ == "__main__":
    config = load_baseline_config()
    print(f"模型路径: {config['model_path']}")
    print(f"LoRA rank: {config['lora']['rank']}")
```

---

## 三、基准测试脚本

### 3.1 主测试脚本

```python
#!/usr/bin/env python3
"""
scripts/benchmark.py - MLX-LoRA 基准测试脚本

功能:
1. 运行多种配置的训练
2. 收集性能指标
3. 生成基准报告
4. 推荐时间预算
"""

import os
import time
import json
import psutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List
import subprocess

import mlx.core as mx
from mlx_lm import load

@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    config_name: str
    config: Dict

    # 时间指标
    setup_time: float      # 模型加载时间
    train_time: float      # 训练时间
    eval_time: float       # 评估时间
    total_time: float      # 总时间

    # 性能指标
    train_loss: float      # 最终训练 loss
    val_loss: float        # 验证 loss
    val_accuracy: float    # 验证准确率
    perplexity: float      # 困惑度

    # 资源指标
    peak_memory_mb: float  # 峰值内存
    avg_memory_mb: float   # 平均内存
    gpu_utilization: float # GPU 利用率

    # 训练效率
    tokens_per_second: float
    steps_per_second: float
    samples_trained: int

class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, output_dir: str = "outputs/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkMetrics] = []

    def get_memory_usage(self) -> float:
        """获取当前内存使用 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def run_single_benchmark(
        self,
        config_name: str,
        config: Dict
    ) -> BenchmarkMetrics:
        """运行单个基准测试"""

        print(f"\n{'='*60}")
        print(f"🔬 运行基准测试: {config_name}")
        print(f"{'='*60}")
        print(f"配置: {json.dumps(config, indent=2)}")

        # 记录开始状态
        start_memory = self.get_memory_usage()
        peak_memory = start_memory
        total_start = time.time()

        # 1. 模型加载计时
        print("\n📦 加载模型...")
        setup_start = time.time()

        model, tokenizer = load(config["model"])

        setup_time = time.time() - setup_start
        print(f"   模型加载时间: {setup_time:.2f}s")

        # 2. 训练计时
        print(f"\n🚀 开始训练 (iters={config['iters']})...")
        train_start = time.time()

        # 运行训练 (这里需要实际的训练代码)
        # 简化版: 调用 train.py
        train_result = self._run_training(config)

        train_time = time.time() - train_start

        # 3. 评估计时
        print(f"\n📊 评估中...")
        eval_start = time.time()

        eval_result = self._run_evaluation(config)

        eval_time = time.time() - eval_start

        # 4. 汇总结果
        total_time = time.time() - total_start

        metrics = BenchmarkMetrics(
            config_name=config_name,
            config=config,
            setup_time=setup_time,
            train_time=train_time,
            eval_time=eval_time,
            total_time=total_time,
            train_loss=train_result.get("train_loss", 0),
            val_loss=eval_result.get("val_loss", 0),
            val_accuracy=eval_result.get("val_accuracy", 0),
            perplexity=eval_result.get("perplexity", 0),
            peak_memory_mb=train_result.get("peak_memory_mb", 0),
            avg_memory_mb=train_result.get("avg_memory_mb", 0),
            gpu_utilization=train_result.get("gpu_utilization", 0),
            tokens_per_second=train_result.get("tokens_per_second", 0),
            steps_per_second=train_result.get("steps_per_second", 0),
            samples_trained=train_result.get("samples_trained", 0),
        )

        # 打印结果
        self._print_metrics(metrics)

        return metrics

    def _run_training(self, config: Dict) -> Dict:
        """运行训练 (调用 train.py)"""

        # 构建命令行参数
        args = [
            "python", "train.py",
            "--lora-rank", str(config.get("lora_rank", 8)),
            "--lora-alpha", str(config.get("lora_alpha", 16)),
            "--batch-size", str(config.get("batch_size", 4)),
            "--iters", str(config.get("iters", 100)),
            "--learning-rate", str(config.get("learning_rate", 5e-5)),
            "--max-seq-length", str(config.get("max_seq_length", 512)),
        ]

        # 运行训练
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )

        # 解析输出
        metrics = self._parse_train_output(result.stdout)

        return metrics

    def _run_evaluation(self, config: Dict) -> Dict:
        """运行评估"""
        # 调用评估脚本
        args = ["python", "scripts/evaluate_adapter.py"]

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        return self._parse_eval_output(result.stdout)

    def _parse_train_output(self, output: str) -> Dict:
        """解析训练输出"""
        metrics = {}

        for line in output.split('\n'):
            if "val_loss:" in line:
                metrics["val_loss"] = float(line.split(":")[1].strip())
            elif "train_loss:" in line:
                metrics["train_loss"] = float(line.split(":")[1].strip())
            elif "peak_memory_mb:" in line:
                metrics["peak_memory_mb"] = float(line.split(":")[1].strip())
            elif "tokens_per_second:" in line:
                metrics["tokens_per_second"] = float(line.split(":")[1].strip())

        return metrics

    def _parse_eval_output(self, output: str) -> Dict:
        """解析评估输出"""
        metrics = {}
        for line in output.split('\n'):
            if "val_loss:" in line:
                metrics["val_loss"] = float(line.split(":")[1].strip())
            elif "val_accuracy:" in line:
                metrics["val_accuracy"] = float(line.split(":")[1].strip())
            elif "perplexity:" in line:
                metrics["perplexity"] = float(line.split(":")[1].strip())
        return metrics

    def _print_metrics(self, metrics: BenchmarkMetrics):
        """打印指标"""
        print(f"\n{'='*60}")
        print(f"📊 测试结果: {metrics.config_name}")
        print(f"{'='*60}")

        print(f"\n⏱️  时间指标:")
        print(f"   模型加载:   {metrics.setup_time:.2f}s")
        print(f"   训练时间:   {metrics.train_time:.2f}s")
        print(f"   评估时间:   {metrics.eval_time:.2f}s")
        print(f"   总时间:     {metrics.total_time:.2f}s")

        print(f"\n📈 性能指标:")
        print(f"   Train Loss: {metrics.train_loss:.6f}")
        print(f"   Val Loss:   {metrics.val_loss:.6f}")
        print(f"   Accuracy:   {metrics.val_accuracy:.4f}")
        print(f"   Perplexity: {metrics.perplexity:.2f}")

        print(f"\n💾 资源使用:")
        print(f"   峰值内存:   {metrics.peak_memory_mb:.0f} MB")
        print(f"   平均内存:   {metrics.avg_memory_mb:.0f} MB")

        print(f"\n⚡ 训练效率:")
        print(f"   Tokens/s:   {metrics.tokens_per_second:.0f}")
        print(f"   Steps/s:    {metrics.steps_per_second:.2f}")

    def run_benchmark_suite(self, configs: Dict[str, Dict]) -> Dict:
        """运行完整的基准测试套件"""

        print(f"\n{'='*60}")
        print(f"🎯 MLX-LoRA 基准测试套件")
        print(f"{'='*60}")
        print(f"测试配置数量: {len(configs)}")
        print(f"预计总时间: ~{len(configs) * 15} 分钟")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 运行所有配置
        for name, config in configs.items():
            try:
                metrics = self.run_single_benchmark(name, config)
                self.results.append(metrics)
            except Exception as e:
                print(f"❌ 测试失败: {name} - {e}")

        # 生成报告
        return self.generate_report()

    def generate_report(self) -> Dict:
        """生成基准测试报告"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "machine_info": self._get_machine_info(),
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary(),
            "recommendations": self._generate_recommendations()
        }

        # 保存报告
        report_path = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # 打印总结
        self._print_summary(report)

        return report

    def _get_machine_info(self) -> Dict:
        """获取机器信息"""
        import platform
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        }

    def _generate_summary(self) -> Dict:
        """生成总结统计"""
        if not self.results:
            return {}

        baseline = self.results[0]  # 第一个是基线

        return {
            "baseline": {
                "config_name": baseline.config_name,
                "val_loss": baseline.val_loss,
                "train_time": baseline.train_time,
                "peak_memory_mb": baseline.peak_memory_mb,
            },
            "time_budget_recommendation": self._recommend_time_budget(),
            "memory_constraint": self._recommend_memory_constraint(),
            "performance_range": {
                "min_val_loss": min(r.val_loss for r in self.results),
                "max_val_loss": max(r.val_loss for r in self.results),
                "avg_train_time": sum(r.train_time for r in self.results) / len(self.results),
            }
        }

    def _recommend_time_budget(self) -> Dict:
        """推荐时间预算"""
        if not self.results:
            return {"time_budget_minutes": 15, "reason": "默认值"}

        avg_time = sum(r.train_time for r in self.results) / len(self.results)
        max_time = max(r.train_time for r in self.results)

        # 时间预算 = 平均时间 + 20% 缓冲
        recommended = max_time * 1.2 / 60  # 转换为分钟

        return {
            "time_budget_minutes": round(recommended, 1),
            "avg_train_time_minutes": round(avg_time / 60, 1),
            "max_train_time_minutes": round(max_time / 60, 1),
            "reason": f"基于 {len(self.results)} 次测试的平均值"
        }

    def _recommend_memory_constraint(self) -> Dict:
        """推荐内存约束"""
        if not self.results:
            return {"max_memory_gb": 16, "reason": "默认值"}

        max_memory = max(r.peak_memory_mb for r in self.results)
        # 约束 = 峰值 + 20% 缓冲
        recommended_gb = (max_memory * 1.2) / 1024

        return {
            "max_memory_gb": round(recommended_gb, 1),
            "peak_memory_gb": round(max_memory / 1024, 1),
            "reason": f"基于 {len(self.results)} 次测试的峰值"
        }

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if not self.results:
            return recommendations

        baseline = self.results[0]

        # 性能分析
        if baseline.val_loss > 2.0:
            recommendations.append("⚠️  基线 val_loss 较高，建议增加训练迭代或调整学习率")

        # 时间分析
        if baseline.train_time > 600:  # 10分钟
            recommendations.append("⏱️  训练时间较长，考虑减少 iters 或 batch_size")

        # 内存分析
        if baseline.peak_memory_mb > 14000:  # 14GB
            recommendations.append("💾 内存使用较高，建议启用 grad_checkpoint 或降低 batch_size")

        # 对比分析
        best = min(self.results, key=lambda r: r.val_loss)
        if best.config_name != "baseline":
            recommendations.append(f"✅ 最佳配置是 '{best.config_name}' (val_loss={best.val_loss:.4f})")

        return recommendations

    def _print_summary(self, report: Dict):
        """打印总结"""
        print(f"\n{'='*60}")
        print(f"📊 基准测试总结")
        print(f"{'='*60}")

        summary = report["summary"]

        print(f"\n🎯 基线性能:")
        b = summary["baseline"]
        print(f"   Val Loss:    {b['val_loss']:.6f}")
        print(f"   训练时间:    {b['train_time']/60:.2f} 分钟")
        print(f"   峰值内存:    {b['peak_memory_mb']:.0f} MB")

        print(f"\n⏱️  推荐时间预算:")
        tb = summary["time_budget_recommendation"]
        print(f"   建议时间预算: {tb['time_budget_minutes']} 分钟")
        print(f"   理由: {tb['reason']}")

        print(f"\n💾 推荐内存约束:")
        mc = summary["memory_constraint"]
        print(f"   建议最大内存: {mc['max_memory_gb']} GB")

        print(f"\n📈 性能范围:")
        pr = summary["performance_range"]
        print(f"   Val Loss:    {pr['min_val_loss']:.6f} ~ {pr['max_val_loss']:.6f}")
        print(f"   平均训练时间: {pr['avg_train_time']/60:.2f} 分钟")

        print(f"\n💡 优化建议:")
        for rec in report["recommendations"]:
            print(f"   {rec}")


# 测试配置 (从 model_config.yaml 读取模型路径)
import yaml
from pathlib import Path

def get_model_path():
    """获取本地模型路径"""
    config_path = Path("config/model_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config["model"]["path"]
    return "models/Qwen3.5-9B-MLX-4bit"  # 默认路径

MODEL_PATH = get_model_path()

BENCHMARK_CONFIGS = {
    "baseline": {
        "model": MODEL_PATH,  # 使用本地模型
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 4,
        "iters": 100,
        "learning_rate": 5e-5,
        "max_seq_length": 512,
        "optimizer": "adam",
        "grad_checkpoint": True,
    },
    "fast_test": {
        "model": MODEL_PATH,  # 使用本地模型
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "batch_size": 8,
        "iters": 50,
        "learning_rate": 5e-5,
        "max_seq_length": 256,
        "optimizer": "adam",
        "grad_checkpoint": True,
    },
    "large_rank": {
        "model": MODEL_PATH,  # 使用本地模型
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "batch_size": 4,
        "iters": 100,
        "learning_rate": 5e-5,
        "max_seq_length": 512,
        "optimizer": "adam",
        "grad_checkpoint": True,
    },
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX-LoRA 基准测试")
    parser.add_argument("--config", choices=["all", "baseline", "fast"], default="baseline",
                        help="测试配置")
    parser.add_argument("--output-dir", default="outputs/benchmarks",
                        help="输出目录")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (只运行 baseline)")

    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output_dir)

    if args.quick or args.config == "baseline":
        # 只运行基线
        configs = {"baseline": BENCHMARK_CONFIGS["baseline"]}
    elif args.config == "fast":
        configs = {"baseline": BENCHMARK_CONFIGS["baseline"],
                   "fast_test": BENCHMARK_CONFIGS["fast_test"]}
    else:
        configs = BENCHMARK_CONFIGS

    # 运行基准测试
    report = runner.run_benchmark_suite(configs)

    # 保存路径
    print(f"\n✅ 基准测试完成!")
    print(f"📄 报告已保存到: {runner.output_dir}")


if __name__ == "__main__":
    main()
```

### 3.2 快速基线测试

```python
#!/usr/bin/env python3
"""
scripts/quick_baseline.py - 快速基线测试

用于快速验证系统是否正常工作，并获取基础指标。
"""

import time
import subprocess
import sys
from pathlib import Path

def run_quick_baseline():
    """运行快速基线测试"""

    print("🚀 快速基线测试")
    print("="*60)

    # 检查数据
    if not Path("data/mlx_lm/train.jsonl").exists():
        print("❌ 数据不存在，请先运行: python prepare.py")
        sys.exit(1)

    # 检查模型
    if not Path("models/Qwen3.5-9B-MLX-4bit").exists():
        print("❌ 模型不存在，请下载模型")
        sys.exit(1)

    print("\n✅ 环境检查通过")

    # 运行训练
    print("\n📊 运行基线训练...")
    start = time.time()

    result = subprocess.run(
        ["python", "train.py"],
        capture_output=False
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✅ 基线测试完成")
        print(f"⏱️  总耗时: {elapsed/60:.2f} 分钟")

        # 解析结果
        with open("run.log") as f:
            for line in f:
                if "val_loss:" in line:
                    print(f"📈 Val Loss: {line.split(':')[1].strip()}")
    else:
        print(f"\n❌ 训练失败")
        sys.exit(1)


if __name__ == "__main__":
    run_quick_baseline()
```

---

## 四、基准测试流程

### 4.1 完整测试流程

```
┌─────────────────────────────────────────────────────────────┐
│                    基准测试流程                              │
└─────────────────────────────────────────────────────────────┘

    Step 1: 环境检查 (5分钟)
    ├─ 检查模型文件
    ├─ 检查训练数据
    ├─ 检查 MLX 安装
    └─ 检查内存可用性

    Step 2: 快速验证 (10分钟)
    ├─ 运行 fast_test 配置
    ├─ 验证系统正常
    └─ 获取初步数据

    Step 3: 基线测试 (15分钟)
    ├─ 运行 baseline 配置
    ├─ 记录所有指标
    └─ 建立性能基线

    Step 4: 参数影响测试 (30分钟)
    ├─ large_rank (大秩)
    ├─ long_seq (长序列)
    ├─ deep_train (深度训练)
    └─ 收集参数影响数据

    Step 5: 报告生成 (5分钟)
    ├─ 汇总所有数据
    ├─ 生成图表
    ├─ 推荐配置
    └─ 保存报告
```

### 4.2 需要收集的数据

| 类别 | 指标 | 用途 |
|------|------|------|
| **时间** | setup_time | 模型加载时间参考 |
| | train_time | 设置时间预算 |
| | eval_time | 评估耗时参考 |
| **性能** | train_loss | 监控训练稳定性 |
| | val_loss | 主要优化目标 |
| | perplexity | 模型质量指标 |
| **资源** | peak_memory_mb | 内存约束设置 |
| | avg_memory_mb | 资源规划参考 |
| | gpu_utilization | 硬件利用分析 |
| **效率** | tokens_per_second | 吞吐量评估 |
| | steps_per_second | 训练速度参考 |

---

## 五、基准报告格式

### 5.1 报告结构

```json
{
  "timestamp": "2026-03-30T14:30:00",
  "machine_info": {
    "platform": "macOS-14.5-arm64-arm-64bit",
    "processor": "arm",
    "memory_gb": 32.0,
    "mlx_version": "0.18.0"
  },
  "results": [
    {
      "config_name": "baseline",
      "train_time": 542.3,
      "val_loss": 1.234567,
      "peak_memory_mb": 12345.6,
      ...
    }
  ],
  "summary": {
    "baseline": {
      "val_loss": 1.234567,
      "train_time_minutes": 9.0
    },
    "time_budget_recommendation": {
      "time_budget_minutes": 12.0,
      "reason": "基于 4 次测试的最大值 + 20% 缓冲"
    },
    "memory_constraint": {
      "max_memory_gb": 16.0,
      "peak_memory_gb": 13.5
    }
  },
  "recommendations": [
    "✅ 最佳配置是 'large_rank' (val_loss=1.198234)",
    "💾 内存使用较高，建议启用 grad_checkpoint"
  ]
}
```

### 5.2 人工报告模板

```markdown
# 基准测试报告

**测试时间**: 2026-03-30 14:30
**机器配置**: M2 Max, 32GB RAM
**MLX 版本**: 0.18.0

## 测试结果总览

| 配置 | Val Loss | 训练时间 | 峰值内存 | Tokens/s |
|------|----------|----------|----------|----------|
| baseline | 1.2346 | 9.0min | 12.3GB | 1234 |
| fast_test | 1.3124 | 4.5min | 10.1GB | 2345 |
| large_rank | 1.1982 | 12.3min | 14.8GB | 987 |

## 推荐配置

**时间预算**: 12 分钟/实验
**内存约束**: 16 GB
**目标 Val Loss**: < 1.20 (较基线改进 3%)

## 优化建议

1. 使用 large_rank 配置作为起点
2. 内存充足，可以尝试 rank=32
3. 训练时间合理，当前 iters=100 适中

## 下一步

基于基准结果，开始自主实验循环。
```

---

## 六、执行计划更新

### 6.1 新增阶段 0：基准测试

在原有执行计划前增加：

```
阶段 0 (1天): 基准测试
├── 环境检查 (30分钟)
├── 快速验证 (1小时)
├── 基线测试 (2小时)
├── 参数影响测试 (3小时)
└── 报告生成 (30分钟)

输出:
- benchmark_report.json
- 推荐的 time_budget
- 推荐的 memory_constraint
- 基线 val_loss 值
```

### 6.2 更新后的时间线

```
阶段 0 (1天)  →  阶段一 (1-2天)  →  阶段二 (2-3天)  →  阶段三 (3-5天)  →  阶段四 (2-3天)
    │                 │                   │                   │                   │
    ▼                 ▼                   ▼                   ▼                   ▼
  基准测试           基础搭建           评估决策             优化增强             验证部署
(建立基线)         (MVP系统)        (实验循环)          (生产级)            (可发布)
```

---

## 七、使用指南

### 7.1 快速开始

```bash
# 1. 准备数据
python prepare.py

# 2. 快速基线测试 (10分钟)
python scripts/quick_baseline.py

# 3. 完整基准测试 (1小时)
python scripts/benchmark.py

# 4. 查看报告
cat outputs/benchmarks/benchmark_*.json
```

### 7.2 集成到主流程

基准测试完成后，将结果写入配置：

```yaml
# config/benchmark_results.yaml
benchmark:
  timestamp: "2026-03-30T14:30:00"
  machine: "M2 Max 32GB"

  # 基线指标
  baseline:
    val_loss: 1.234567
    perplexity: 15.23
    train_time_minutes: 9.0

  # 推荐配置
  recommendations:
    time_budget_minutes: 12.0
    max_memory_gb: 16.0
    target_val_loss: 1.20

  # 参数影响
  parameter_impact:
    lora_rank:
      low: 1.3124      # rank=4
      baseline: 1.2346 # rank=8
      high: 1.1982     # rank=16
    batch_size:
      small: 1.2456    # batch=2
      baseline: 1.2346 # batch=4
      large: 1.2389    # batch=8
```

---

**文档版本**: 1.0
**最后更新**: 2026-03-30
**状态**: 待审核
