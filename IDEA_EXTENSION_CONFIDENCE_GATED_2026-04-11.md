# Idea Extension Report

日期：2026-04-11  
语言：中文  
扩展方向：`Confidence-Gated Selective Invariance`

## 1. 为什么要再想一个新 idea

第二轮结果已经给出一个比较清楚的信号：

- `selected-neg` 比 `selected-no-neg` 略好
- 但提升很小
- 同时整体 accuracy 明显下降

这意味着当前主线的问题不是“完全没效果”，而是：

> 一旦把 invariance pressure 提高到能显出 `negative control` 的差异，模型整体行为也开始变差。

所以本轮不再提出“大换题”的新 backbone，而是提出一个更小、更可证伪的新 extension：

`Confidence-Gated Selective Invariance`

## 2. 核心想法

不是所有训练样本都需要同样强的 latent invariance。

直觉上：

- 对已经很确定、answer loss 很低的样本，再强压 orbit regularization，容易变成多余约束
- 对高不确定性样本，latent alignment 才更可能真的有帮助

所以新的想法是：

- 保留原有 `selected-no-neg` / `selected-neg` mask
- 但不再对所有 batch 样本均匀施加 orbit loss
- 改为按样本 answer NLL 生成一个 `gate weight`
- 只让高不确定性样本承担更大的 orbit 正则权重

## 3. 为什么这不算“泛 router”

这个 extension 故意不做 generic hybrid router，也不引入新的 latent/text backbone。

它仍然是当前 selective invariance 主线的最小扩展：

- selector 不变
- 数据不变
- backbone 不变
- 只是把 `who gets regularized more` 从“所有样本一样”改成“更难样本更多”

因此它更像：

- `training-time gating`

而不是：

- `test-time hybrid routing`

## 4. 最小实现

已在 `scripts/train_selective_invariance.py` 中实现：

- 新增 variants：
  - `full-state-gated`
  - `selected-no-neg-gated`
  - `selected-neg-gated`
- 新增逻辑：
  - 从 main / soc branch 的 token-level answer NLL 计算每个样本的平均难度
  - 用 `sigmoid((nll - gate_center) / gate_temperature)` 生成 gate weight
  - 用 gate weight 对 per-example orbit loss 加权
- 新增日志：
  - `train_gate_weight_mean`
  - summary 中的 `gate.enabled / center / temperature`

本轮正式实验使用：

- `gate_center = 1.5`
- `gate_temperature = 0.5`

## 5. 实验设置

与第二轮加压对照 matched：

- backbone：`Llama-3.2-1B-Instruct`
- train pairs：`512`
- scoring pairs：`128`
- eval pairs：`128`
- epochs：`2`
- `lambda_orbit = 0.1`

对比：

- 旧：`selected-no-neg`
- 旧：`selected-neg`
- 新：`selected-no-neg-gated`
- 新：`selected-neg-gated`

## 6. 结果

| Variant | Mean Acc | Rewrite Consistency | Acc-Conditioned Rewrite Consistency |
|---|---:|---:|---:|
| `selected-no-neg` | `0.8125` | `0.7891` | `0.7344` |
| `selected-neg` | `0.8164` | `0.8047` | `0.7422` |
| `selected-no-neg-gated` | `0.8242` | `0.7812` | `0.7266` |
| `selected-neg-gated` | `0.8242` | `0.7812` | `0.7266` |

对应汇总文件：

- `pilot_results/llama1b_round3_gated_compare.csv`

## 7. 如何解读

这组结果说明：

1. gated 确实改变了训练行为，因为平均准确率有所恢复。
2. 但它没有保住主指标上的优势，`accuracy-conditioned rewrite consistency` 反而下降。
3. 更关键的是，`selected-neg-gated` 没有再优于 `selected-no-neg-gated`。

所以目前最准确的结论不是：

- `gated is better`

而是：

- `gated trades some robustness pressure for better average accuracy`

## 8. 结论

`Confidence-Gated Selective Invariance` 目前的状态是：

- `IMPLEMENTED`
- `EXPERIMENTALLY TESTED`
- `NOT PROMOTED`

它可以保留为后续小 sweep 的备选扩展，但不能替代当前主线，也不能作为新的主论文 route。
