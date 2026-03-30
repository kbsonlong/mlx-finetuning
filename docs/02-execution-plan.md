# MLX-LoRA 自主微调系统 - 执行计划

> **创建日期**: 2026-03-30
> **预计周期**: 11-15 天 (含基准测试)

---

## 执行概览

```
阶段0 (1天)  →  阶段一 (1-2天)  →  阶段二 (2-3天)  →  阶段三 (3-5天)  →  阶段四 (2-3天)
    │                 │                    │                    │                    │
    ▼                 ▼                    ▼                    ▼                    ▼
  基准测试          基础搭建             评估决策              优化增强              验证部署
(建立基线)        (MVP系统)         (实验循环)           (生产级)            (可发布)
```

---

## 阶段 0：基准测试 (Day 1)

### 目标
建立性能基线，确定合理的时间预算和优化目标

**详细方案**: 请参阅 [基准测试方案](./04-benchmark-plan.md)

### 关键任务

| 任务 | 预计时间 | 说明 |
|------|----------|------|
| 环境检查 | 30分钟 | 验证 MLX、模型、数据就绪 |
| 快速验证 | 1小时 | 运行 fast_test 配置验证系统 |
| 基线测试 | 2小时 | 运行 baseline 配置获取基线 |
| 参数影响测试 | 3小时 | 测试不同配置对性能的影响 |
| 报告生成 | 30分钟 | 分析结果，确定时间预算等 |

### 核心产出

```yaml
# config/benchmark_results.yaml
benchmark:
  baseline:
    val_loss: 1.234567        # 基线性能
    train_time_minutes: 9.0   # 训练耗时

  recommendations:
    time_budget_minutes: 12.0 # 基于测试确定
    max_memory_gb: 16.0       # 基于峰值内存
    target_val_loss: 1.20     # 优化目标
```

### 验收标准

- [ ] 所有测试配置成功运行
- [ ] 生成完整基准报告 (JSON)
- [ ] 确定时间预算 (基于实际数据)
- [ ] 确定内存约束 (基于峰值 + 缓冲)
- [ ] 建立清晰性能基线

---

## 阶段一：基础搭建 (Days 2-3)

### 目标
创建可运行的最小可行系统 (MVP)

### 任务清单

#### 1.1 项目初始化 [Day 2 上午]

- [ ] 创建项目目录结构
```bash
mlx-autoresearch/
├── prepare.py
├── train.py
├── program.md
├── pyproject.toml
├── scripts/
├── config/
└── outputs/
```

- [ ] 初始化 Git 仓库
```bash
git init
git checkout -b main
cat > .gitignore << 'EOF'
outputs/
runs/
.pytest_cache/
__pycache__/
*.pyc
*.npz
*.safetensors
EOF
```

- [ ] 创建 `pyproject.toml`
```toml
[project]
name = "mlx-autoresearch"
version = "0.1.0"
dependencies = [
    "mlx>=0.18.0",
    "mlx-lm>=0.19.0",
    "torch>=2.0.0",
    "litellm>=1.0.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
    "pyarrow>=14.0.0",
]

[tool.uv]
dev-dependencies = []
```

**预计时间**: 1 小时

#### 1.2 数据准备模块 [Day 1 下午]

- [ ] 移植 `deepseek_distill.py`
  - [ ] DeepSeek API 调用
  - [ ] 成本估算
  - [ ] 进度显示

- [ ] 移植 `augment_distilled.py`
  - [ ] 上下文增强
  - [ ] 复杂度增强
  - [ ] 去重逻辑

- [ ] 创建 `prepare.py`
  - [ ] 数据下载/生成入口
  - [ ] MLX 格式转换
  - [ ] Train/Valid/Test 分割

**预计时间**: 4 小时

#### 1.3 基础训练脚本 [Day 2 上午]

- [ ] 实现 `train.py` 核心逻辑
```python
# 关键组件
├── 配置加载 (YAML)
├── 模型加载 (MLX-LM)
├── LoRA 配置
├── 训练循环
├── 验证评估
└── 结果输出
```

- [ ] 实现超参数接口
```python
# train.py 顶部可配置区
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
ITERS = 100
MAX_SEQ_LENGTH = 512
```

**预计时间**: 4 小时

#### 1.4 AI 指令集 [Day 2 下午]

- [ ] 编写 `program.md` 初版
  - [ ] 实验设置说明
  - [ ] 可修改范围定义
  - [ ] 决策规则
  - [ ] 循环指令

**预计时间**: 2 小时

#### 1.5 MVP 测试 [Day 2 下午]

- [ ] 准备测试数据 (小规模)
- [ ] 运行一次完整训练
- [ ] 验证输出格式
- [ ] 确认所有组件连通

**预计时间**: 2 小时

### 阶段一产出

- [ ] 可运行的训练脚本
- [ ] 准备好的测试数据
- [ ] 基础 AI 指令集
- [ ] MVP 测试报告

### 验收标准

1. `python prepare.py` 能生成 MLX 格式数据
2. `python train.py` 能完成训练并输出结果
3. 输出包含 `val_loss` 等关键指标

---

## 阶段二：评估与决策 (Days 3-5)

### 目标
实现完整的实验循环系统

### 任务清单

#### 2.1 独立评估脚本 [Day 3]

- [ ] 创建 `scripts/evaluate_adapter.py`
```python
# 功能
├── 加载训练好的适配器
├── 在验证集上推理
├── 计算评估指标
│   ├── loss
│   ├── accuracy
│   └── perplexity
└── 格式化输出
```

- [ ] 实现多种评估模式
  - [ ] 快速评估 (采样)
  - [ ] 完整评估 (全量)
  - [ ] 增量评估 (新数据)

**预计时间**: 3 小时

#### 2.2 结果格式化 [Day 3]

- [ ] 统一输出格式
```python
# train.py 结尾输出
print("---")
print(f"val_loss:        {val_loss:.6f}")
print(f"val_accuracy:    {val_accuracy:.4f}")
print(f"perplexity:      {perplexity:.2f}")
print(f"train_time:      {train_time:.1f}")
print(f"peak_memory_mb:  {peak_memory:.1f}")
print(f"lora_rank:       {LORA_RANK}")
print(f"learning_rate:   {LEARNING_RATE}")
print(f"batch_size:      {BATCH_SIZE}")
```

- [ ] 实现 `grep` 友好格式
  - 每个指标独占一行
  - 格式稳定可解析

**预计时间**: 2 小时

#### 2.3 结果记录系统 [Day 4]

- [ ] 实现 JSONL 记录
```python
# outputs/results.jsonl 格式
# 每行一个完整的 JSON 对象
{
  "experiment_id": "exp_20260330_001",
  "timestamp": "2026-03-30T14:30:00Z",
  "config": { ... },
  "metrics": {
    "val_loss": 1.234567,
    "perplexity": 45.67,
    ...
  },
  "status": "completed",
  "branch": "exp/exp_20260330_001"
}
```

- [ ] 创建自动记录逻辑
```python
def register_experiment_result(experiment_id, config, metrics, status, branch):
    """记录实验结果到 JSONL"""
    result = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "status": status,
        "branch": branch
    }
    with open("outputs/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
```

- [ ] 添加结果查询工具
```python
# scripts/query_results.py
def get_best_results(n=5):
    """获取最好的 N 个结果"""
    results = load_all_results()
    sorted_results = sorted(results, key=lambda x: x["metrics"]["val_loss"])
    return sorted_results[:n]
```

**预计时间**: 3 小时

#### 2.4 实验分支管理 [Day 4]

- [ ] 实验分支创建
```python
def create_experiment_branch(experiment_id: str):
    """为每次实验创建独立分支"""
    branch_name = f"exp/{experiment_id}"
    subprocess.run(["git", "checkout", "-b", branch_name])
    return branch_name

def commit_experiment(description: str):
    """提交实验修改"""
    subprocess.run(["git", "add", "train.py"])
    result = subprocess.run(
        ["git", "commit", "-m", description],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def discard_experiment():
    """放弃当前实验（删除实验分支）"""
    current_branch = get_current_branch()
    subprocess.run(["git", "checkout", "main"])
    subprocess.run(["git", "branch", "-D", current_branch])

def promote_to_main(experiment_branch: str):
    """将成功的实验合并到主分支"""
    subprocess.run(["git", "checkout", "main"])
    subprocess.run(["git", "merge", "--no-ff", experiment_branch, "-m",
                    f"Promote: {experiment_branch}"])

def get_current_branch():
    """获取当前分支名"""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()
```

- [ ] 分支策略
  - [ ] 主分支: `main` - 存放最佳配置
  - [ ] 实验分支: `exp/{experiment_id}` - 每次实验独立分支
  - [ ] 失败实验直接删除分支
  - [ ] 成功实验合并到 main

- [ ] 结果登记
```python
# 所有实验结果记录到 outputs/results.jsonl
# 无论实验成功或失败，都记录完整信息
def register_experiment_result(config, metrics, status, branch):
    result = {
        "experiment_id": config["experiment_id"],
        "config": config,
        "metrics": metrics,
        "status": status,  # success, failed, timeout
        "branch": branch,
        "timestamp": datetime.now().isoformat()
    }
    with open("outputs/results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
```

**预计时间**: 4 小时

#### 2.5 决策逻辑 [Day 5]

- [ ] 实现决策函数
```python
def should_keep_new_result(new_val_loss, best_val_loss, threshold=0.01):
    """决定是否保留新结果"""
    return new_val_loss < best_val_loss - threshold
```

- [ ] 添加简化奖励
```python
def complexity_score(config):
    """计算配置复杂度 (越低越好)"""
    return (config['lora_rank'] *
            config['lora_alpha'] *
            config['batch_size'])
```

**预计时间**: 2 小时

### 阶段二产出

- [ ] 独立评估脚本
- [ ] 统一输出格式
- [ ] 结果记录系统 (outputs/results.jsonl)
- [ ] 实验分支管理系统
- [ ] 决策逻辑

### 验收标准

1. 能独立评估已训练的适配器
2. 结果能被正确解析和记录到 JSONL
3. 实验分支自动创建和管理
4. 失败实验安全丢弃，不影响主分支
5. 决策逻辑正确执行

---

## 阶段三：优化与增强 (Days 6-10)

### 目标
将系统提升到生产级质量

### 任务清单

#### 3.1 超参数搜索策略 [Day 6]

- [ ] 实现随机搜索
```python
def random_search(space, n_samples=10):
    """从参数空间随机采样"""
    samples = []
    for _ in range(n_samples):
        config = {k: random.choice(v) for k, v in space.items()}
        samples.append(config)
    return samples
```

- [ ] 实现网格搜索 (可选)
- [ ] 实现贝叶斯优化 (高级)

**预计时间**: 4 小时

#### 3.2 早停机制 [Day 6]

- [ ] 监控训练 loss
```python
if train_loss > prev_loss * 1.5:  # 发散检测
    print("Training diverged, stopping early")
    break
```

- [ ] 验证集 patience
```python
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter > max_patience:
        break
```

**预计时间**: 2 小时

#### 3.3 实验可视化 [Day 7]

- [ ] 结果图表生成
```python
# scripts/plot_results.py
├── val_loss 趋势图
├── 参数分布图
└── 并行坐标图
```

- [ ] 使用 matplotlib/plotly

**预计时间**: 4 小时

#### 3.4 并行实验 [Day 8]

- [ ] 多进程实验调度
```python
def run_parallel_experiments(configs, n_workers=2):
    """并行运行多个配置"""
    with Pool(n_workers) as pool:
        pool.map(run_single_experiment, configs)
```

- [ ] 资源管理 (避免 OOM)

**预计时间**: 4 小时

#### 3.5 异常恢复 [Day 9]

- [ ] 检查点保存
```python
# 每 N 步保存检查点
if step % save_every == 0:
    save_checkpoint(model, optimizer, f"ckpt_{step}")
```

- [ ] 崩溃恢复
```python
# 从检查点恢复
if os.path.exists("latest_ckpt"):
    model, optimizer = load_checkpoint("latest_ckpt")
```

- [ ] 实验状态持久化

**预计时间**: 4 小时

#### 3.6 优化 program.md [Day 10]

- [ ] 完善 AI 指令集
  - [ ] 添加更多探索策略
  - [ ] 添加调试指令
  - [ ] 添加边界情况处理

**预计时间**: 3 小时

### 阶段三产出

- [ ] 超参数搜索工具
- [ ] 早停机制
- [ ] 可视化脚本
- [ ] 并行实验支持
- [ ] 异常恢复机制
- [ ] 优化的 AI 指令集

### 验收标准

1. 能进行高效的参数搜索
2. 训练异常能正确处理
3. 实验结果可直观可视化
4. 系统稳定运行不崩溃

---

## 阶段四：验证与部署 (Days 11-14)

### 目标
确保系统可发布使用

### 任务清单

#### 4.1 端到端测试 [Day 11]

- [ ] 完整循环测试
  - [ ] 10+ 次连续实验
  - [ ] 验证决策正确性
  - [ ] 检查资源释放

- [ ] 边界测试
  - [ ] 空数据集
  - [ ] 极端参数值
  - [ ] 内存限制

**预计时间**: 4 小时

#### 4.2 性能基准 [Day 11]

- [ ] 与人工调参对比
```python
# 相同数据集，比较:
1. 人工最优配置
2. 自主研究最优配置
3. 随机配置
```

- [ ] 记录性能数据

**预计时间**: 3 小时

#### 4.3 文档编写 [Day 12]

- [ ] README.md
  - [ ] 项目介绍
  - [ ] 快速开始
  - [ ] 使用指南
  - [ ] API 文档

- [ ] 部署指南
- [ ] 故障排查
- [ ] FAQ

**预计时间**: 5 小时

#### 4.4 持续运行测试 [Days 13-14]

- [ ] 长时间稳定性测试
  - [ ] 100+ 次实验循环
  - [ ] 监控内存泄漏
  - [ ] 检查 Git 增长

- [ ] 性能回归测试
- [ ] 多场景验证

**预计时间**: 8 小时

### 阶段四产出

- [ ] 测试报告
- [ ] 性能基准数据
- [ ] 完整文档
- [ ] 稳定性验证

### 验收标准

1. 通过所有测试用例
2. 性能优于人工调参
3. 文档完整清晰
4. 长时间运行稳定

---

## 总体进度跟踪

| 阶段 | 任务 | 预计时间 | 状态 | 备注 |
|------|------|----------|------|------|
| 0 | 环境检查 | 0.5h | ⬜ | |
| 0 | 快速验证 | 1h | ⬜ | |
| 0 | 基线测试 | 2h | ⬜ | |
| 0 | 参数影响测试 | 3h | ⬜ | |
| 0 | 报告生成 | 0.5h | ⬜ | |
| 一 | 项目初始化 | 1h | ⬜ | |
| 一 | 数据准备模块 | 4h | ⬜ | |
| 一 | 基础训练脚本 | 4h | ⬜ | |
| 一 | AI 指令集 | 2h | ⬜ | |
| 一 | MVP 测试 | 2h | ⬜ | |
| 二 | 独立评估 | 3h | ⬜ | |
| 二 | 结果格式化 | 2h | ⬜ | |
| 二 | 结果记录 | 3h | ⬜ | |
| 二 | Git 集成 | 3h | ⬜ | |
| 二 | 决策逻辑 | 2h | ⬜ | |
| 三 | 搜索策略 | 4h | ⬜ | |
| 三 | 早停机制 | 2h | ⬜ | |
| 三 | 可视化 | 4h | ⬜ | |
| 三 | 并行实验 | 4h | ⬜ | |
| 三 | 异常恢复 | 4h | ⬜ | |
| 三 | 指令优化 | 3h | ⬜ | |
| 四 | 端到端测试 | 4h | ⬜ | |
| 四 | 性能基准 | 3h | ⬜ | |
| 四 | 文档编写 | 5h | ⬜ | |
| 四 | 稳定性测试 | 8h | ⬜ | |

---

## 风险管理

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| MLX 版本兼容 | 高 | 中 | 锁定版本，预留适配时间 |
| 数据质量 | 高 | 低 | 充分测试数据准备 |
| 资源不足 | 中 | 中 | 优化 batch size，梯度累积 |
| API 配额 | 低 | 低 | 本地缓存，减少调用 |

---

## 依赖项

### 外部依赖

- [ ] MLX >= 0.18.0
- [ ] MLX-LM >= 0.19.0
- [ ] DeepSeek API Key
- [ ] Qwen3.5-9B-MLX 模型

### 内部依赖

- [ ] 基础 Python 环境
- [ ] Git 配置
- [ ] 足够磁盘空间 (50GB+)

---

## 审核检查清单

在开始执行前，请确认:

- [ ] 已阅读并理解集成方案
- [ ] 技术路线确认可行
- [ ] 资源准备充分
- [ ] 时间计划可接受
- [ ] 风险缓解措施明确

**审核通过后可开始执行阶段一**

---

**文档版本**: 1.0
**最后更新**: 2026-03-30
**状态**: 待审核
