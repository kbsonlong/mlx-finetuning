# MLX-LoRA 基准测试报告

> 测试日期: 2026-03-30
> 执行环境: 本地 `python3` (`~/.pyenv/versions/mlx/bin/python3`) + MLX/Metal
> 数据版本: `ds_20260330_145725`
> 模型: `Qwen3.5-9B-MLX-4bit`

---

## 一、执行摘要

本次重新运行了 `scripts/benchmark.py` 的 6 组预设，全部基于真实 MLX 训练完成，且已经使用修复后的解析逻辑重新生成正式报告。

核心结论:

- 当前最佳配置是 `small_batch`，`val_loss=0.658333`，耗时 `364.35s`。
- 默认 `baseline` 的真实结果为 `val_loss=0.725000`，耗时 `324.70s`，较旧 heuristic 报表有本质差异。
- `fast_test` 仍然是最快的一组，耗时 `167.27s`，但效果不再是最优，说明当前评估结果存在较高波动。
- `deep_train` 将 `baseline` 的 `val_loss` 从 `0.725000` 降到 `0.740278` 反而变差，且训练时间增至 `634.14s`，当前数据上不划算。
- 最新正式 JSON/YAML 已经是清洁结构，只包含最终摘要字段。
- 当前验证集只有 `1` 条样本，`tokens_evaluated=18`，所以本轮 benchmark 只能作为方向性结论，不能作为稳定的最优配置定论。

---

## 二、测试结果

| preset | val_loss | val_accuracy | perplexity | train_time(s) | train_time(min) | peak_mem(GB) | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| small_batch | 0.658333 | 0.603015 | 1.931570 | 364.35 | 6.07 | 未单独写入报告 | 当前最优 |
| baseline | 0.725000 | 0.579710 | 2.064731 | 324.70 | 5.41 | 未单独写入报告 | 真实基线 |
| deep_train | 0.740278 | 0.574621 | 2.096518 | 634.14 | 10.57 | 未单独写入报告 | 更慢但没有更好 |
| fast_test | 0.791667 | 0.558140 | 2.207072 | 167.27 | 2.79 | 未单独写入报告 | 最快，但效果一般 |
| large_rank | 0.794444 | 0.557276 | 2.213211 | 312.75 | 5.21 | 未单独写入报告 | 与 fast_test 接近 |
| long_seq | 0.895833 | 0.527473 | 2.449376 | 310.99 | 5.18 | 未单独写入报告 | 当前最差 |

机器生成报告同步写入:

- `config/benchmark_results.yaml`
- `outputs/benchmark_results.json`

---

## 三、结果解读

### 3.1 baseline 已经被修正

本轮真实训练得到的 `baseline` 为:

- `val_loss=0.725000`
- `train_time=324.70s`

这和旧结果中的 `0.159785 / 0.00s` 有本质差异，说明旧报表混入了 heuristic 或非真实训练结果，后续不应再拿旧 benchmark 作为时间预算依据。

### 3.2 训练更久不等于更好

`deep_train` 从 100 iter 增加到 200 iter 后，`train_time` 几乎翻倍，但效果并没有变好:

- baseline: `val_loss=0.725000`, `324.70s`
- deep_train: `val_loss=0.740278`, `634.14s`

说明在当前极小数据集上，继续堆训练步数已经进入明显的收益不稳定区，甚至可能只是增加过拟合噪声。

### 3.3 长序列在当前数据上没有价值

`long_seq` 的 `val_loss=0.895833`，是当前最差的一组，而耗时仍然在 `310.99s`。这说明当前样本非常短，增大 `MAX_SEQ_LENGTH` 到 `1024` 只增加了配置复杂度，没有换来效果提升。

### 3.4 最优点已经发生漂移

上一轮 `fast_test` 曾经最好，但在这轮清洁版 benchmark 中，最优点变成了 `small_batch`:

- `small_batch`: `val_loss=0.658333`
- `baseline`: `val_loss=0.725000`
- `fast_test`: `val_loss=0.791667`

这不是参数搜索已经稳定收敛的信号，反而说明当前验证集太小，最优点会随着单次训练波动而漂移。

### 3.5 显存不是当前瓶颈

从训练日志看，本轮各组训练仍然在约 `6 GB` 量级运行。在 `16 GB` 预算下，真正的瓶颈更像是:

- 数据量过小
- 验证集过小
- 参数变化彼此耦合，难以判断单因子效果

---

## 四、当前 benchmark 的可信度边界

本轮结果能用于指导下一步，但不适合直接固定最终最优配置。原因很明确:

- 训练集只有 `8` 条样本。
- 验证集只有 `1` 条样本。
- `tokens_evaluated=18`，统计波动极大。

这意味着:

- 当前最优 `small_batch` 很可能只是对这一个验证样本拟合得最好。
- `0.658333` 和 `0.725000` 之间的差距未必在更大验证集上还能成立。
- 现在更重要的是先提高 benchmark 信噪比，再谈精细搜索。

---

## 五、优化方向

### 5.1 第一优先级: 扩大验证集，先修评估再修参数

建议先把验证集扩到至少 `20-50` 条样本，再重新跑 benchmark。否则后续搜索会被单样本噪声误导。

建议动作:

- 增加 `valid.jsonl` 样本数，避免只有 1 条数据。
- 保持 `test.jsonl` 独立，不把 test 用作调参依据。
- benchmark 完成后固定对比同一份 frozen 数据版本。

### 5.2 第二优先级: 围绕 small_batch 和 baseline 附近做局部搜索

当前最值得继续验证的区域不是 `fast_test`，而是 `small_batch` 和 `baseline` 周边。

建议下一轮只改一个变量:

- `rank`: `8 / 12 / 16`
- `iters`: `80 / 100 / 150`
- `batch_size`: `2 / 4`
- `max_seq_length`: `256 / 512`

先做局部网格，而不是继续扩大全量搜索空间。

### 5.3 第三优先级: 分离“速度优化”和“效果优化”

目前多个 preset 同时改了多项参数，结论会互相干扰。下一轮建议拆成两条线:

- 速度线: 固定效果参数，只比较 `batch_size`、`seq_length`、是否 grad checkpoint。
- 效果线: 固定吞吐相关参数，只比较 `rank`、`iters`、`learning_rate`。

这样才能看清楚每个参数到底在影响什么。

### 5.4 第四优先级: 重新审视时间预算

目前脚本输出的推荐时间预算是 `15.57` 分钟，这个值来自最慢一次训练再加 5 分钟。就本轮数据而言，更实用的预算应分层:

- 快速筛选预算: `3-4` 分钟/次
- 标准验证预算: `5-7` 分钟/次
- 深度确认预算: `10-11` 分钟/次

这比单一时间预算更适合后续搜索流程。

### 5.5 第五优先级: 当前报告结构已修复

`scripts/benchmark.py` 的解析逻辑已经修复，当前正式 JSON/YAML 只保留最终摘要字段。

当前保留字段包括:

- `experiment_id`
- `dataset_id`
- `mode`
- `preset`
- `val_loss`
- `val_accuracy`
- `perplexity`
- `tokens_evaluated`
- `train_time`
- `lora_rank`
- `learning_rate`
- `batch_size`
- `adapter_file`
- `adapter_path`

这已经足够支持后续自动分析，短期内不需要再动这块。

---

## 六、建议的下一轮实验

推荐直接执行下面这一轮最小闭环:

1. 扩充 `valid.jsonl` 到至少 20 条。
2. 保持 `test.jsonl` 不参与调参。
3. 围绕 `small_batch` 和 `baseline` 做小范围搜索。
4. 优先单独验证 `batch_size` 和 `iters`，避免一次改多个变量。
5. 再次生成 benchmark 报告，与本报告做同口径对比。

---

## 七、附录

本轮 6 个真实 MLX 训练实验对应目录:

- `outputs/adapters/exp_20260330_161218_552691`
- `outputs/adapters/exp_20260330_161745_131362`
- `outputs/adapters/exp_20260330_162034_176700`
- `outputs/adapters/exp_20260330_162548_685813`
- `outputs/adapters/exp_20260330_163101_589773`
- `outputs/adapters/exp_20260330_164137_612799`
