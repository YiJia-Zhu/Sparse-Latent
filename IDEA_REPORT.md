# Idea Discovery Report

方向：`Latent Inference / Hybrid Latent Inference LLM`  
日期：2026-04-11  
语言：中文  
Pipeline：`research-lit -> idea discovery -> novelty framing -> research review -> refine`

## Executive Summary

结合仓库已有调研、现有非训练 pilot，以及 2025-2026 新文献补检，当前最值得继续推进的主线仍然不是“重新发明 latent 架构”，而是：

`Negative-Controlled Selective Invariance for Rewrite Robustness in Latent Reasoning Training`

它的优势是：

- 已有分析信号，不是从零开始
- 和 2026 年的 reviewer 关注点一致，主打 `robustness / confound reduction`，而不是过强的 `faithful reasoning`
- 可自然扩展到 `hybrid latent inference`，作为下一步 backup 方向

## Method Clarification

### 1. 这个 idea 到底在做什么

先用最直白的话说：

> 我们想让模型对“同一道题的不同写法”更稳定，但不要把“答案刚好一样”这种假信号也一起学进去。

这个 idea 不是提出一个全新的 latent backbone。  
它做的是：

1. 在一个现有 backbone 的 hidden state 里，先挑一小部分更值得信的坐标
2. 训练时，不对全状态一刀切，而只对这部分坐标施加 invariance regularization

### 2. 我们筛选的对象是什么

我们筛选的不是题目、不是 token，也不是样本标签。  
我们筛选的是：

- 某个固定层的 hidden state 向量里的单个坐标 / 维度

例如如果 pooled hidden state 的维度是 `d = 2048`，那么我们就在这 `2048` 个坐标里挑出一个固定 top-k mask。

### 3. 为什么要筛选坐标，而不是直接用 full-state

因为 `full-state alignment` 太粗暴。

如果把整段 hidden state 都拿来对齐，很容易把下面这些东西一起对齐进去：

- 表面写法
- 模板差异
- 长度差异
- 题型 identity
- 答案线索

这正是当前主线的核心判断：

> 不是所有 hidden dimensions 都值得被 consistency regularize。

### 4. 为什么需要三件事情一起看

我们不是随便挑坐标，而是希望挑出“更像稳定 reasoning signal”的坐标。  
单看一个条件不够，所以需要三件事情一起约束。

#### 第一件事：同题改写稳定

问题是：

- 同一道题换一种 rationale 写法后，这个坐标是否仍然接近？

作用：

- 排除只对表面措辞敏感的坐标

#### 第二件事：对输出有作用

问题是：

- 这个坐标是否和正确答案 token 的输出有关？

作用：

- 排除虽然稳定、但对最终答案几乎没作用的坐标

#### 第三件事：不要只是“同答案就稳定”

这就是负控。

问题是：

- 如果换成两道不同题，但答案碰巧一样，这个坐标是否也一样稳定？

如果是，那这个坐标更像：

- answer cue

而不是：

- reasoning signal

作用：

- 排除“看起来有用，但其实只是在记答案”的坐标

### 5. `selected-no-neg` 和 `selected-neg` 的区别

- `selected-no-neg`：
  也会筛一组坐标，但只看“同题改写稳定 + 对输出有作用”，不做负控过滤
- `selected-neg`：
  在前两条之外，再用“不同题但同答案”的负控过滤掉 answer-generic 坐标

所以它们的差别不是 backbone，也不是训练预算，而是：

> 是否显式排除 answer cue contamination

### 6. 这个 idea 成立依赖哪些前提

这条路线默认依赖以下前提，如果这些前提不成立，idea 本身就要收缩解释。

1. 同题不同写法确实会在 hidden state 中留下一部分可复现的稳定性信号。
2. 这部分信号不会完全退化成“答案数字”或“题型模板”。
3. 在固定 basis 中，坐标级别的筛选虽然不是唯一真实机制，但仍能提供有用的训练对象。
4. sparse selected-mask regularization 比 full-state regularization 更不容易把噪声一起锁死。
5. 训练后，固定 warm-start mask 虽然可能漂移，但至少足以提供早期有效约束。

### 7. 这条 idea 不在声称什么

为了避免误解，当前版本明确不声称：

- 找到了“真正推理坐标”
- 这些坐标具有唯一、固定、可解释的语义身份
- non-training analysis 已经证明了行为收益

当前更安全的表述是：

> same-answer negative control 可能帮助我们筛出一组更适合做 rewrite invariance regularization 的坐标。

## Literature Landscape

### 1. 现在的 latent reasoning 不是单一路线

补检后的文献版图大致可分为四条：

1. 压缩型 latent CoT  
   代表：Hidden CoT、Continuous Latent Space Reasoning  
   结论：latent 计算是可行的，但未解决 faithful / confound 问题。

2. latent test-time scaling / recurrent depth  
   代表：Recurrent Depth、Looped Transformers  
   结论：latent steps 可以作为额外内部计算预算，但更深不等于更可信。

3. hybrid latent inference  
   代表：Hybrid Latent Reasoning via RL、SPOT  
   结论：latent 与 token 的混合模式已经成为独立研究方向，但 generic router 已开始拥挤。

4. mechanism / reliability / faithfulness 质疑  
   代表：Beyond Chains of Thought benchmark、Do Latent Tokens Think、Do Latent-CoT Models Think Step-by-Step、Dynamics Within Latent CoT、When Shallow Wins  
   结论：以后 latent 论文必须说明内部过程是否更稳、更少 shortcut，不能只报 accuracy。

### 2. 我们最适合站在哪个缝里

对当前仓库和资源约束来说，最自然的结合点不是再造 backbone，而是把 latent reasoning 与 feature selection / SAE / representation engineering 结合起来，回答：

> latent reasoning 训练里，到底应该对哪些内部坐标做 consistency regularization？

这和以下文献形成自然邻接，但还没被直接做死：

- `Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models`
- `SAEs Are Good for Steering -- If You Select the Right Features`
- `Hybrid Latent Reasoning via Reinforcement Learning`
- `Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models`

## Ranked Ideas

### 1. Negative-Controlled Selective Invariance for Rewrite Robustness in Latent Reasoning Training

状态：`RECOMMENDED`

- 核心问题：
  latent reasoning 训练若直接做 full-state alignment，很容易把答案线索、模板、长度噪声也一起对齐进去。
- 核心想法：
  先在 warm-start checkpoint 上，从 pooled hidden state 的单个坐标里筛一组固定 top-k coordinates；筛选时同时要求：
  - 同题改写稳定
  - 对输出有作用
  - 不要只是 same-answer generic signal
  然后训练时只对这组坐标做 rewrite invariance regularization。
- 主张边界：
  不讲“发现了真推理坐标”；只讲 `same-answer negative control helps selective invariance improve rewrite robustness`。
- 当前证据：
  analysis-only pilot 为正；`selected-neg` 在 11/11 个运行中优于 `selected-no-neg`。
- Novelty 判断：
  `CONFIRMED-LEANING`
  目前最接近的是 feature selection / representation consistency / steering，但还没有看到“same-answer negative control + fixed sparse invariance mask + latent reasoning training”这个组合被清楚做成主命题。
- Reviewer 风险：
  如果训练结果不明显优于 `selected-no-neg` 和 `full-state`，项目应停止。
- Pilot 信号：
  `POSITIVE (analysis-only, not training)`
- 推荐原因：
  这是当前唯一同时满足“已有信号、叙事安全、实现规模适中”的主线。

### 2. Confidence-Gated Hybrid Latent Inference with Negative-Controlled Route Regularization

状态：`BACKUP`

- 核心问题：
  hybrid latent inference 已经开始出现，但现有工作更多强调 latent/text 兼容性和策略学习，较少讨论 rewrite robustness 与 shortcut contamination。
- 核心想法：
  在 hybrid latent inference 中，不直接做 generic route policy，而是只在高不确定性片段启用 latent mode，并对 latent route 的 selected coordinates 施加 negative-controlled regularization。
- 与现有工作差异：
  相比 `Hybrid Latent Reasoning via RL`，这里主打的不是 RL policy 本身，而是 `route robustness + latent coordinate cleanliness`。
- Novelty 判断：
  `PLAUSIBLE BUT UNVERIFIED`
- 主要风险：
  需要先有可运行的 hybrid backbone，否则工程量会迅速膨胀。
- 当前证据：
  无 pilot。
- 为什么保留：
  它是最有希望把“纯 latent 主线”平滑扩展到 `hybrid latent inference` 的备选路线。

### 3. Correct-vs-Wrong Latent Separation as a Selector Signal

状态：`BACKUP`

- 核心问题：
  直接找“好的 reasoning coordinate”太难，但正确轨迹和错误轨迹是否可分，可能更容易先验证。
- 核心想法：
  用 correct-vs-wrong separation 作为 selector 的一个附加信号，再服务于后续 regularization，而不是单独做 verifier / reranker。
- Novelty 判断：
  `PARTIAL`
  与 hallucination detection、correctness steering、hidden cognition 邻近。
- 当前价值：
  更像 Idea 1 的辅助模块，而不是独立主论文。

## Eliminated Ideas

### 1. Prototype-Based Dynamic Latent Guidance

- 原因：
  与 2025 年后半的 dynamic steering / feedback controller / prototype steering 过近。
- 结论：
  更适合作为 baseline 或组件，不适合做主线。

### 2. Causal-Necessity Regularization for Latent Steps

- 原因：
  2026 年初已经有多篇工作正面讨论 latent step 的 causal structure 与 faithfulness。
- 结论：
  风险高，且 reviewer 很容易要求远超当前资源的机制证据。

## Gate 1 Result

按照 `research-pipeline` 的默认设置，`AUTO_PROCEED = true`。

本轮自动选择：

`Idea 1 — Negative-Controlled Selective Invariance for Rewrite Robustness in Latent Reasoning Training`

自动选择理由：

- 它是当前唯一有现成 analysis signal 的方向
- 它和已有 reviewer 反馈完全对齐
- 它最容易写成一个小而清晰、能被证伪的主命题

## Refined Proposal

现有 refine 结果与本轮判断一致，可继续复用：

- Proposal：`refine-logs/FINAL_PROPOSAL.md`
- Experiment plan：`refine-logs/EXPERIMENT_PLAN.md`
- Review summary：`RESEARCH_REVIEW_TOP_IDEA_2026-04-11.md`

## Recommended Next Step

如果继续按主线推进，最合理的顺序不是立刻做大规模训练，而是：

1. 先做最小训练对照，只跑三个条件：
   - `augmentation-matched no-reg`
   - `selected-no-neg`
   - `selected-neg`
2. 如果 `selected-neg` 在 held-out rewrite consistency 上仍优于 `selected-no-neg`，再补：
   - `full-state orbit`
   - `random top-k`
3. 若主结果成立，再考虑把方法扩展到 `hybrid latent inference` 场景

## Sources

- https://arxiv.org/abs/2409.08561
- https://arxiv.org/abs/2412.06769
- https://arxiv.org/abs/2502.05171
- https://arxiv.org/abs/2502.17416
- https://arxiv.org/abs/2504.10615
- https://arxiv.org/abs/2505.15634
- https://arxiv.org/abs/2505.16782
- https://arxiv.org/abs/2505.18454
- https://arxiv.org/abs/2512.21711
- https://arxiv.org/abs/2601.08058
- https://arxiv.org/abs/2602.00449
- https://arxiv.org/abs/2602.08783
- https://arxiv.org/abs/2603.03475
- https://arxiv.org/abs/2603.06222
- https://aclanthology.org/2025.emnlp-main.519/
- https://aclanthology.org/2025.naacl-long.264/
