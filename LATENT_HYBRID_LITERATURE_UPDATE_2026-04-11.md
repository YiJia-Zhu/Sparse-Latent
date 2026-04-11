# Latent / Hybrid Latent Inference 文献补充更新

日期：2026-04-11

## 1. 这次补充更新的目的

仓库里已有一版较完整的 latent reasoning 调研。本次补充不是重写一遍，而是把 2025 年中到 2026 年初更关键的几条新线索补齐，回答两个更直接的问题：

1. `latent inference` 这条线到今天为止最真实的前沿分层是什么。
2. 我们如果想做“结合创新”的隐推理方向，哪里还有空间，哪里已经明显拥挤。

## 2. 当前版图：四条已经成形的主线

### 2.1 压缩型 latent CoT

这一类工作的目标是：保留多步推理收益，但不显式输出长 CoT。

- `Hidden CoT`：把完整 CoT 压缩成更短的隐式中间过程，核心卖点是降低推理时延，同时尽量保留 reasoning 增益。
  - https://arxiv.org/abs/2409.08561
- `Training LLMs to Reason in a Continuous Latent Space`：明确提出在连续隐空间而不是语言空间里进行中间计算。
  - https://arxiv.org/abs/2412.06769

这条线已经证明了一件事：`latent reasoning` 不是空想，它确实可以在不显式吐出完整 rationale 的情况下带来收益。

但它留下来的核心空白也很明显：

- 提分不等于过程可信
- 压缩进去的东西可能混有答案线索、模板、捷径

### 2.2 test-time latent scaling / recurrent depth

2025 年开始，latent reasoning 不再只是“压缩 CoT”，而开始被当成一种 test-time compute 扩展方式。

- `Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach`
  - https://arxiv.org/abs/2502.05171
- `Reasoning with Latent Thoughts: On the Power of Looped Transformers`
  - https://arxiv.org/abs/2502.17416

这一阶段的启发是：

- latent step 可以当作额外内部计算预算
- 深度循环可能比继续输出 token 更划算

但这也直接把后面的 reviewer 问题暴露了出来：

- step 变多是不是只是在“多算”，而不是“算对”
- latent depth 的提升是否真的对应更好的 reasoning quality

### 2.3 hybrid latent inference 开始成形

这一类工作最值得重视，因为它直接说明“纯 latent”未必是唯一方向，`latent + token` 的混合模式已经开始成为一条独立路线。

- `Hybrid Latent Reasoning via Reinforcement Learning`
  - https://arxiv.org/abs/2505.18454
- `SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models`
  - https://arxiv.org/abs/2603.06222

这两篇合起来说明：

- 纯连续 latent 推理和离散 autoregressive 生成之间的冲突，已经被视为一个值得单独优化的问题
- hybrid 不是简单混搭，而是在找“什么时候该隐想，什么时候该显式吐 token”
- 如果我们后面想做 `Hybrid latent inference`，新意不能只是“也做混合”，而必须回答一个更细的问题：
  - 如何更稳地决定 latent/text 的切换或约束对象
  - 如何降低 hybrid 模式下的 shortcut / answer cue 污染

### 2.4 faithful / mechanism / reliability 质疑成为主流

到 2025 年末和 2026 年初，领域重心已经明显转向：

> latent reasoning 能提分，但它到底是不是在“真推理”？

关键论文包括：

- `Beyond Chains of Thought: Benchmarking Latent-Space Reasoning Abilities in Large Language Models`
  - https://arxiv.org/abs/2504.10615
- `Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought`
  - https://arxiv.org/abs/2512.21711
- `Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks`
  - https://arxiv.org/abs/2602.00449
- `Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure`
  - https://arxiv.org/abs/2602.08783
- `When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning`
  - https://arxiv.org/abs/2603.03475

这批工作共同推出来的边界非常重要：

- 不能再只报 accuracy
- 不能默认 latent steps 就是 faithful steps
- 更不能轻易宣称“发现了真正 reasoning neurons / coordinates”

这意味着我们后面的论文叙事必须更保守：

- 可以讲 `robustness`
- 可以讲 `consistency`
- 可以讲 `confound reduction`
- 不要把主张写成 `faithful reasoning discovered`

## 3. 另一条可结合的线：feature / SAE / steering

如果只看 latent reasoning 主线，会误以为解决方案必须来自新架构。但 2025-2026 的 feature / SAE / steering 文献其实给了我们更现实的结合点。

- `Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models`
  - https://arxiv.org/abs/2505.15634
- `SAEs Are Good for Steering -- If You Select the Right Features`
  - https://aclanthology.org/2025.emnlp-main.519/
- `Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering`
  - https://aclanthology.org/2025.naacl-long.264/
- `Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models`
  - https://arxiv.org/abs/2601.08058

这条线给我们的不是“已经解决 latent reasoning”，而是三个更实际的启发：

1. hidden state 里确实有可用结构，不必默认只能全状态蒸馏。
2. 不是所有 feature 都值得约束，必须先做筛选。
3. feature 选择如果不控制 confound，很容易学到答案线索而不是推理结构。

## 4. 对我们当前方向最关键的三条判断

### 4.1 最稳的创新点不在“再造 latent 架构”

单纯新造一个 latent backbone，今天已经没有明显性价比。现阶段更合理的是：

- 在已有 latent / hybrid latent backbone 上
- 加一个小但清楚的训练对象或筛选原则
- 把故事讲成 `where to regularize` 或 `when to use latent mode`

### 4.2 `Hybrid latent inference` 现在有价值，但必须避开“只是再做一个路由器”

`Hybrid Latent Reasoning via RL` 已经把“latent 和 discrete generation 的兼容性”放到台面上。  
如果我们直接做一个 latent/text router，很容易被问：

- 这和已有 hybrid RL 有什么本质不同？
- 你的贡献是 routing policy，还是 latent robustness 本身？

所以更自然的切入不是 generic hybrid policy，而是：

- hybrid 模式下的 rewrite robustness
- hybrid 模式下 latent/text route consistency
- hybrid 模式下 negative-controlled feature selection

### 4.3 纯 latent 方向最现实的切口是“选择性约束”，不是“证明真推理”

当前最合理的主线叙事应该是：

> latent reasoning training 不应该对全状态一刀切地做 consistency；应该先用更干净的 selector 找到更值得约束的 latent coordinates，再施加 selective invariance。

这条叙事：

- 与现有 reviewer 反馈一致
- 与 analysis-only 结果一致
- 与 2025-2026 的文献趋势一致

## 5. 对我们有价值的论文分组清单

### A. 必读主线

1. Hidden CoT
   - https://arxiv.org/abs/2409.08561
2. Continuous Latent Space Reasoning
   - https://arxiv.org/abs/2412.06769
3. Recurrent Depth
   - https://arxiv.org/abs/2502.05171
4. Looped Transformers
   - https://arxiv.org/abs/2502.17416
5. Latent CoT Survey
   - https://arxiv.org/abs/2505.16782

### B. 必读质疑线

1. Beyond Chains of Thought benchmark
   - https://arxiv.org/abs/2504.10615
2. Do Latent Tokens Think?
   - https://arxiv.org/abs/2512.21711
3. Do Latent-CoT Models Think Step-by-Step?
   - https://arxiv.org/abs/2602.00449
4. Dynamics Within Latent CoT
   - https://arxiv.org/abs/2602.08783
5. When Shallow Wins
   - https://arxiv.org/abs/2603.03475

### C. 必读结合线

1. Hybrid Latent Reasoning via RL
   - https://arxiv.org/abs/2505.18454
2. SPOT
   - https://arxiv.org/abs/2603.06222
3. Feature Extraction and Steering for CoT
   - https://arxiv.org/abs/2505.15634
4. SAEs Are Good for Steering
   - https://aclanthology.org/2025.emnlp-main.519/
5. Reasoning Beyond CoT as latent computational mode
   - https://arxiv.org/abs/2601.08058

## 6. 结论

到 2026-04-11 为止，`latent / hybrid latent inference` 的前沿已经不是“latent 能不能 work”，而是三件更细的事：

1. latent 过程是不是足够稳、足够可解释、足够少 shortcut
2. hybrid 模式下何时 latent、何时 text，能不能更有原则
3. consistency / regularization 应该施加在什么对象上，而不是对全状态一刀切

因此，最适合我们当前资源与已有积累的方向，不是重新发明一个大而全的 latent architecture，而是做：

`negative-controlled selective invariance`

或它在 `hybrid latent inference` 下的更窄、更强、更可证伪的版本。
