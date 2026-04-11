# Research Pipeline Report

方向：`Latent Inference / Hybrid Latent Inference LLM`  
选择的 idea：`Negative-Controlled Selective Invariance for Rewrite Robustness in Latent Reasoning Training`  
日期：2026-04-11  
Pipeline：`idea-discovery -> implement -> run-experiment -> auto-review-loop`

## Journey Summary

- 已完成：
  - 读取并整合仓库既有调研、候选 idea、reviewer 反馈、refine proposal、experiment plan
  - 补检 2025-2026 的 latent / hybrid latent inference 关键论文
  - 重新排序 ideas，并自动选择最稳主线
  - 在第二轮弱正结果基础上，额外提出并实现了一个最小 extension：
    `Confidence-Gated Selective Invariance`
- 当前判断：
  - 最强主线仍是 `negative-controlled selective invariance`
  - `hybrid latent inference` 仍更适合作为第二阶段扩展，而不是当前主论文的起点
  - 新的 gated extension 已经实验验证，但暂时不能升级为新的主线

## Stage Status

### Stage 1: Idea Discovery

状态：`DONE`

产物：

- `LATENT_HYBRID_LITERATURE_UPDATE_2026-04-11.md`
- `IDEA_REPORT.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`

### Stage 2: Implementation

状态：`DONE (minimum package + one tested extension)`

已完成的最小实现包：

- `scripts/train_selective_invariance.py`
- `scripts/summarize_selective_invariance_runs.py`
- 基于本地 `Llama-3.2-1B-Instruct` checkpoint 的 LoRA 训练 / eval
- JSON / CSV 结果落盘
- 最小四条件包运行完成

本轮新增的 extension 实现：

- `full-state-gated`
- `selected-no-neg-gated`
- `selected-neg-gated`

gated 机制：

- 用 main / soc prompt 的 detached answer NLL 估计样本不确定性
- 只对高不确定性样本施加更高的 orbit regularization 权重

实现层面仍然缺的不是“能不能跑”，而是：

- 更强的 held-out rewrite 构造
- 更大训练预算下的稳定 run 计划
- 与 `CODI` / 真 latent backbone 更紧的接法

### Stage 3: Run Experiment

状态：`INITIAL RESULTS READY + EXTENSION TESTED`

已经完成：

1. warm-start baseline
2. `aug-no-reg`
3. `full-state`
4. `selected-no-neg`
5. `selected-neg`
6. 第二轮 budget-lift 对照：`selected-no-neg` vs `selected-neg`
7. 第三轮 extension 对照：`selected-no-neg-gated` vs `selected-neg-gated`

第三轮关键信息：

- 设置与第二轮 matched：
  - `train_pairs=512`
  - `scoring_pairs=128`
  - `eval_pairs=128`
  - `epochs=2`
  - `lambda_orbit=0.1`
- 结果：
  - `selected-no-neg-gated`
    - `mean_accuracy = 0.8242`
    - `rewrite_consistency = 0.7812`
    - `acc_conditioned_rewrite_consistency = 0.7266`
  - `selected-neg-gated`
    - `mean_accuracy = 0.8242`
    - `rewrite_consistency = 0.7812`
    - `acc_conditioned_rewrite_consistency = 0.7266`
- 解释：
  - gated 版本相比第二轮 nongated，恢复了一部分平均准确率
  - 但主指标 `accuracy-conditioned rewrite consistency` 下降
  - `selected-neg-gated` 没有再拉开 `selected-no-neg-gated`
  - 因此 gated extension 目前只支持“缓解 accuracy tradeoff 的尝试”，不支持“增强 negative control 主命题”

### Stage 4: Auto Review Loop

状态：`EARLY STOP AFTER INTERNAL REVIEW`

原因：

- 当前结果仍是 `inconclusive / weak positive`
- 新 extension 没有把核心 claim 拉强
- 已完成一轮内部 reviewer-style 复盘，但不值得进入高成本的多轮 auto-review

## Final Status

- [ ] Ready for submission
- [x] Direction locked and reframed
- [x] Top idea selected
- [x] Experiment story fixed
- [x] Minimal training package implemented
- [x] One follow-up idea experimentally validated
- [ ] Main result obtained

## Current Best Reading

- 目前最好的行为结果仍是第二轮 nongated：
  - `selected-neg` 略优于 `selected-no-neg`
  - 但优势很弱，而且伴随整体性能下降
- 第三轮 gated extension 说明：
  - “把 orbit loss 只压在高不确定样本”这个想法是可运行的
  - 但它没有保住主指标上的弱优势
  - 所以不能把它当作新的 winning route

## Decision Gate

当前判定：

- 不进入 full paper / submission push
- 不进入高成本 auto-review-loop
- 保留主线与 extension 代码，作为后续低成本复现实验的基础

判停条件仍然不变：

- 如果后续多 seed 或轻扫 `lambda_orbit / gate_center` 后，仍然不能稳定得到
  `selected-neg > selected-no-neg` 且不明显掉准确率，
  就不应继续扩展成 full paper

## Remaining TODOs

- 复现一个不掉准确率的 `selected-neg > selected-no-neg`
- 如果要继续测试 gated 路线，只做极小 sweep：
  - `gate_center`
  - `gate_temperature`
  - `lambda_orbit`
- 加上多 seed
- 做更独立的 held-out rewrite
- 决定是否继续主线或降级主张

## Files Changed

- `LATENT_HYBRID_LITERATURE_UPDATE_2026-04-11.md`
- `IDEA_REPORT.md`
- `IDEA_EXTENSION_CONFIDENCE_GATED_2026-04-11.md`
- `AUTO_REVIEW.md`
- `RESEARCH_PIPELINE_REPORT.md`
- `scripts/train_selective_invariance.py`
- `scripts/summarize_selective_invariance_runs.py`
- `refine-logs/EXPERIMENT_RESULTS.md`
