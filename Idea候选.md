# 纯 LLM Latent 推理：多个可做 Idea 整理

**目标**: 不再只保留一个主线，而是把当前最像样的 latent 推理 idea 拆成多个独立点。  
**范围**: 纯 LLM，不做多模态，不依赖大规模预训练，优先考虑 `1B-8B` 开源模型、LoRA / SFT 级可实现方案。  

---

## 2026-04 当前结论版

这一版是在重新检查本地调研材料，并补看一批 2025-2026 近作之后给出的。

### 本地资源约束

- 已有数据集：`gsm8k`、`MATH-500`、`AIME_2024`、`amc23`、`ai2_arc`
- 当前 `huggingface_models` 目录是空的，因此真正开跑 pilot 之前，还需要补一个小模型；最现实的是 `1B-2B` 量级指令模型
- 所以 idea 排名会更偏向：小样本、少改模型、能先做分析和轻量正则的路线

### 快速查新后的总体判断

我重点补看了这些与当前想法最接近的论文：

- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)
- [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)
- [CorrSteer: Steering Improves Task Performance and Safety in LLMs through Correlation-based Sparse Autoencoder Feature Selection](https://arxiv.org/abs/2508.12535)
- [SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)
- [Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)
- [Prototype-Based Dynamic Steering for Large Language Models](https://arxiv.org/abs/2510.05498)
- [Eliciting Chain-of-Thought in Base LLMs via Gradient-Based Representation Optimization](https://arxiv.org/abs/2511.19131)
- [Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](https://arxiv.org/abs/2507.10007)

基于这些近作，当前更稳的判断是：

- `Idea 1` 不是直接撞车，但 novelty 依赖你是否把“orbit 一致性 + feature 选择”真正做成一个训练目标，而不只是分析
- `Idea 2` 很稳，但与 feature selection / hidden-state regularization 邻近工作距离最近，容易被 reviewer 说“这是更稳妥的 engineering 变体”
- `Idea 3` 和已有 `PDS / IDS / PID steering` 太近，更适合作为 baseline 或组件，不适合当前当主线
- `Idea 4` 更像启动模块，不像主论文
- `Idea 5` 仍然有空间，尤其如果你把它从“检测”推进到“用于 latent regularization 或 reranking 的训练信号”
- `Idea 6` 研究味最强，但风险大，因为 2025-2026 已经有多篇论文在做 latent causal diagnostics

### 当前推荐排序

| 排名 | Idea | 新颖性 | 可行性 | 风险 | 当前判断 |
|---|---|---|---|---|---|
| 1 | `Feature-Selected Rationale-Orbit Regularization` | `高-中` | `中` | `中` | 最像主线 |
| 2 | `Correct-vs-Wrong Latent Separation for Reasoning` | `中-高` | `中` | `中` | 最像“先分析后增强”的稳路线 |
| 3 | `Latent Feature Regularization` | `中` | `高` | `中低` | 最容易先做 pilot |
| 4 | `Causal-Necessity Regularization for Latent Steps` | `中` | `低-中` | `高` | 研究味强，但最容易被质疑 |
| 5 | `One-Shot / Small-Data Reasoning Direction Bootstrapping` | `中低` | `高` | `中` | 更适合当组件 |
| 6 | `Prototype-Based Dynamic Latent Guidance` | `中低` | `中` | `中` | 和 steering 近作太近 |

### 如果只选 3 个值得继续推进

#### 1. `Feature-Selected Rationale-Orbit Regularization`

- 最核心的差异点是：不是对整段 hidden state 做 consistency，而是只对筛选出的 reasoning-related feature 做 orbit consistency
- 近邻工作虽然很多，但我还没看到“等价 rationale 不变性 + feature selection + latent reasoning faithful signal”被直接合在一起
- reviewer 最可能的质疑是：你这个是不是只是 paraphrase consistency；所以后面实验必须证明约束到的不是语言模板，而是更稳定的推理结构

#### 2. `Correct-vs-Wrong Latent Separation for Reasoning`

- 这条线比直接增强 latent 推理更稳，因为它先问：正确和错误轨迹能不能在 hidden feature 上分开
- 它和 [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)、[Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](https://arxiv.org/abs/2507.10007) 有邻近关系
- 但如果你把它推进成“分离后用于 feature regularization / reranking / early stopping”，仍然有明确方法空间

#### 3. `Latent Feature Regularization`

- 这条线最保守，但最好起步
- 风险不在于能不能做，而在于 reviewer 可能觉得“只是把 steering 里的 feature selection 用到 latent training 里”
- 如果做这条线，最好把卖点放在：`faithful latent supervision`，而不是“又一个 steering 变体”

### 当前不建议做主线的两个方向

#### `Prototype-Based Dynamic Latent Guidance`

- 和 [Prototype-Based Dynamic Steering for Large Language Models](https://arxiv.org/abs/2510.05498)、[In-Distribution Steering](https://arxiv.org/abs/2510.13285)、[PID Steering](https://arxiv.org/abs/2510.04309) 太接近
- 更适合后续做 test-time baseline，而不是主论文

#### `One-Shot / Small-Data Reasoning Direction Bootstrapping`

- 很实用，但更像前处理模块
- 单独拿出来写主论文，容易被问：真正的新东西在哪

### 如果按“现在就能做 pilot”的顺序

1. `Latent Feature Regularization`
2. `Correct-vs-Wrong Latent Separation for Reasoning`
3. `Feature-Selected Rationale-Orbit Regularization`

原因很简单：

- `Idea 2` 对数据要求最低
- `Idea 5` 可以直接用对/错样本构造分析
- `Idea 1` 最有论文相，但前提是你要先有一套比较干净的 orbit 构造方式

### 查新结果：每个 idea 对应的撞车 / 近邻论文

这一节不是说“已经完全被做过”，而是记录每个 idea 最可能被 reviewer 拿来对比、质疑的论文。

#### Idea 1：Feature-Selected Rationale-Orbit Regularization

最接近的撞车 / 近邻论文：

- [Contrastive Instruction Tuning](https://arxiv.org/abs/2402.11138)  
  重合点：都在做“语义等价文本的内部表示一致性”
- [Representation Consistency for Accurate and Coherent LLM Answer Aggregation](https://arxiv.org/abs/2506.21590)  
  重合点：都强调内部表示 consistency 可作为 reasoning 质量信号
- [SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)  
  重合点：都强调不是所有 hidden feature 都值得约束，必须先选 feature
- [Do Sparse Autoencoders Identify Reasoning Features in Language Models?](https://arxiv.org/abs/2601.05679)  
  重合点：都和“reasoning feature 是否真的可信”直接相关

最容易被质疑的点：

- 你是不是只是把 `representation consistency + feature selection` 拼在一起
- 你这个 orbit 是不是本质上只是 paraphrase augmentation

#### Idea 2：Latent Feature Regularization

最接近的撞车 / 近邻论文：

- [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)  
  重合点：都从 CoT / activation 中提取 reasoning-related feature
- [CorrSteer](https://arxiv.org/abs/2508.12535)  
  重合点：都强调 correctness-related feature selection
- [Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)  
  重合点：都强调 noisy vector / noisy feature 需要 refinement
- [Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models](https://arxiv.org/abs/2601.08058)  
  重合点：都在把 reasoning 从显式文本转成 latent computation 的 shaping 问题

最容易被质疑的点：

- 这是不是只是把 steering 里的 feature selection 搬到了训练阶段
- 这更像一个组件，而不是一篇独立论文

#### Idea 3：Prototype-Based Dynamic Latent Guidance

最接近的撞车 / 近邻论文：

- [Prototype-Based Dynamic Steering for Large Language Models](https://arxiv.org/abs/2510.05498)  
  重合点：几乎正面撞题，都是 prototype-based dynamic guidance
- [In-Distribution Steering: Balancing Control and Coherence in Language Model Generation](https://arxiv.org/abs/2510.13285)  
  重合点：都强调 steering / guidance 强度要动态调整
- [Activation Steering with a Feedback Controller](https://arxiv.org/abs/2510.04309)  
  重合点：都在做动态、闭环式 latent/activation 控制

最容易被质疑的点：

- 这是不是就是 PDS 的 latent reasoning 改写版

#### Idea 4：One-Shot / Small-Data Reasoning Direction Bootstrapping

最接近的撞车 / 近邻论文：

- [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)  
  重合点：都强调少量标注样本也可以 bootstrap latent signal
- [CorrSteer](https://arxiv.org/abs/2508.12535)  
  重合点：都在做低数据条件下的 latent feature 提取
- [Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)  
  重合点：都在处理“小数据 latent signal 很脏”的问题

最容易被质疑的点：

- 这是不是只是一个数据效率更高的初始化技巧
- 能不能撑起主论文，而不只是辅助模块

#### Idea 5：Correct-vs-Wrong Latent Separation for Reasoning

最接近的撞车 / 近邻论文：

- [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)  
  重合点：都在做“好/坏 latent 表示分离”
- [Search-Based Correction of Reasoning Chains Improves Long-Term Planning in LLMs](https://arxiv.org/abs/2505.11824)  
  重合点：都和正确/错误 reasoning path 的区分有关
- [Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](https://arxiv.org/abs/2507.10007)  
  重合点：都利用 hidden cognition / hidden state 和可靠 reasoning 的关系
- [CorrSteer](https://arxiv.org/abs/2508.12535)  
  重合点：都把 correctness 作为 feature 筛选信号

最容易被质疑的点：

- 这是不是 latent veracity / correctness separation 的 reasoning 版本
- 如果最后只是做 verifier 或 reranker，就不够新

#### Idea 6：Causal-Necessity Regularization for Latent Steps

最接近的撞车 / 近邻论文：

- [Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)  
  重合点：都在研究 latent step 的 causal structure / necessity
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)  
  重合点：都在问 latent step 是否真的承担 sequential reasoning
- [Towards Generalizable Reasoning: Group Causal Counterfactual Policy Optimization](https://arxiv.org/abs/2602.06475)  
  重合点：都把 counterfactual / causal 干预引入 reasoning 训练
- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)  
  重合点：都在批判 latent step 的“看起来像推理”问题

最容易被质疑的点：

- 这是不是把 causal diagnostics 很自然地搬成 regularizer
- 扰动不等于真正识别因果必要性

---

## 术语先解释

- `activation（激活）`：模型在某一层、某一个 token 位置上的中间数值状态。很多论文直接分析 `residual activation / hidden state`，看这些中间状态和正确性、行为、推理模式的关系，例如 [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)、[Prototype-Based Dynamic Steering for Large Language Models](https://arxiv.org/abs/2510.05498)。
- `feature（特征）`：不是单个神经元，而是 activation 空间里一个更有功能意义的方向、成分或模式。很多 SAE / steering 论文关心的就是“哪些 feature 会影响输出，哪些只是跟输入一起出现”，例如 [Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](https://aclanthology.org/2025.acl-long.1139/)、[SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)。
- `SAE feature`：用 Sparse Autoencoder 把复杂 activation 拆成一组更稀疏、更可分析的成分后得到的特征。它不是“真正的推理原子”已经被证明了，但它确实常被用来做 feature 选择、去噪和 steering，例如 [CorrSteer](https://arxiv.org/abs/2508.12535)、[Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)。

这几个术语和 latent 推理的关系是：

- latent 方法如果直接监督整段 hidden state，很容易把“真正有用的推理信号”和“模板、长度、措辞噪声”一起学进去；
- SAE / steering 相关工作给出的启发不是“已经找到了真正推理单元”，而是“hidden state 里确实存在一些比别的成分更稳定、更相关、更值得约束的结构信号”，这正好能帮助我们设计更稳的 latent 方法 [CorrSteer](https://arxiv.org/abs/2508.12535)、[SAE-RSV](https://arxiv.org/abs/2509.23799)、[SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)。

---

## Idea 1：Feature-Selected Rationale-Orbit Regularization

### 相关论文怎么做

这条 idea 主要站在两类论文上：

1. `Rationale-Orbit Consistency` 相关想法  
核心思想是：
- 同一道题可能有多种等价 rationale
- 一个好的 latent reasoner 不该因为推理写法变了，内部表示就完全乱掉

这里不是指已经有一篇标准同名论文，而是把两类证据拼起来形成一个训练假设：
- 一类工作指出 latent 推理里确实存在“过程看起来很多步，但不一定真的在用这些步”的问题，因此不能只看最终分数，要看内部过程是否稳定、是否必要，例如 [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)、[Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)、[Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)。
- 另一类工作说明中间表征不是完全黑箱，某些 feature 可以被筛出来、干预、去噪，因此“只对一部分更可信的内部结构做一致性约束”是有根据的，例如 [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)、[CorrSteer](https://arxiv.org/abs/2508.12535)。

2. `SAE / 可解释性 / steering` 相关工作  
这些论文告诉我们：
- 不是所有被激活的 hidden feature 都值得信，很多 feature 只是跟输入共现，未必真正影响输出 [SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)、[CorrSteer](https://arxiv.org/abs/2508.12535)
- 更合理的做法是先找出更可信、更稳定、更有输出影响的 feature，再去做控制或约束 [Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](https://aclanthology.org/2025.acl-long.1139/)、[Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)
- reasoning 相关 signal 可以直接从 CoT 或 residual activation 中近似提取出来，不一定必须全空间对齐 [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)

所以，已有工作的两个问题是：
- 单独做 orbit consistency 时，容易退化成“只是在逼不同措辞更像”，最后约束到的是语言表面而不是推理结构；latent faithful reasoning 诊断类工作已经提醒我们，表面上像在推理，不等于内部真的在做必要计算 [Do Latent Tokens Think?](https://arxiv.org/abs/2512.21711)、[Dynamics Within Latent Chain-of-Thought](https://arxiv.org/abs/2602.08783)
- 单独做 latent regularization 时，又说不清到底该 regularize 哪些 feature；SAE/steering 工作则说明 feature selection 比“一把抓整个 hidden state”更关键 [SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)、[CorrSteer](https://arxiv.org/abs/2508.12535)、[Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)

### 这个 idea 怎么改进

把两条线合起来：

- 不是让整段 hidden state 在等价 rationale 下完全一致
- 而是先筛出少量更可信的 reasoning-related feature
- 再只在这些 feature 上做 orbit 一致性约束

也就是说：

`不是“所有内部表示都一样”，而是“真正和高质量推理有关的那部分核心特征更稳定”。`

### 解决什么问题

主要解决两个问题：

1. 模型只记住某一种 teacher rationale 的表面写法  
2. latent regularization 过于粗暴，把有用差异和噪声一起绑死

### 可能取得什么效果

如果做得好，预期效果包括：

- 对等价推理写法更稳
- 更少依赖 teacher rationale 的表面形式
- latent 表示更像抽象推理结构，而不是文本模板
- 在不显著增加计算量的前提下，提高稳定性和泛化

### 实现难度

**中等，可做。**

最小版本可以这样做：
- 选一个小模型
- 选一小类任务
- 每道题构造 2-3 个等价 rationale
- 先找少量特征，再加一致性约束

### 风险

- orbit 构造不干净会导致训练信号变脏
- feature 选得不好会退化成“模板一致性”

### 关键参考文献

- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)
- [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)
- [CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](https://arxiv.org/abs/2508.12535)
- [SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)
- [Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](https://aclanthology.org/2025.acl-long.1139/)
- [Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)

---

## Idea 2：Latent Feature Regularization

### 相关论文怎么做

这条线主要来自：

1. `SAE-RSV`、`CorrSteer`、`SAEs Are Good for Steering -- If You Select the Right Features`  
这些工作都在强调：
- raw steering vector 往往很脏，需要做 feature refinement / denoising [Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)
- 不是所有 feature 都有用，重要的是筛出真正与目标行为相关的 feature [CorrSteer](https://arxiv.org/abs/2508.12535)、[SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)
- feature selection 和 feature denoising 往往比“直接加一个向量”更关键 [Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](https://aclanthology.org/2025.acl-long.1139/)、[Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)

2. 一些 latent reasoning 方法  
这些方法常见问题是：
- 直接对整段 hidden state 做蒸馏或对齐
- 很难知道学到的是 reasoning、答案线索，还是 shortcut；这正是近两批 latent 机制诊断论文持续质疑的问题 [Do Latent Tokens Think?](https://arxiv.org/abs/2512.21711)、[Do Latent-CoT Models Think Step-by-Step?](https://arxiv.org/abs/2602.00449)

### 这个 idea 怎么改进

把重点放在：

`先找 reasoning-related latent feature，再只对这些 feature 做轻量 regularization。`

不是监督整个 hidden state，也不是要求整个 latent trajectory 长得像 teacher，而是：

- 找少量 feature
- 对这些 feature 加约束
- 让它们更稳定、更可分、更少噪声

### 解决什么问题

主要解决：

1. 传统 latent 蒸馏太粗  
2. hidden state 太纠缠，直接对齐副作用大  
3. 模型可能学到和答案、长度、模板相关的伪特征

### 可能取得什么效果

如果做得好，可能带来：

- 更干净的 latent supervision
- 更稳定的训练
- 比整段状态对齐更好的泛化
- 在小数据下也能获得较稳定的改进

### 实现难度

**中等偏低，是最保守、最容易起步的方向之一。**

因为它不要求：
- 大规模 orbit 数据
- test-time search
- 新架构设计

### 风险

- 最难的是证明你选到的 feature 确实值得 regularize
- reviewer 会问：这些 feature 是不是只是相关，而不是机制上真的重要

### 关键参考文献

- [CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](https://arxiv.org/abs/2508.12535)
- [Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement](https://arxiv.org/abs/2509.23799)
- [SAEs Are Good for Steering -- If You Select the Right Features](https://aclanthology.org/2025.emnlp-main.519/)
- [Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](https://aclanthology.org/2025.acl-long.1139/)
- [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)
- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)

---

## Idea 3：Prototype-Based Dynamic Latent Guidance

### 相关论文怎么做

这条线主要参考：

1. `Prototype-Based Dynamic Steering (PDS)`  
做法是：
- 从 CoT prompt 和 neutral prompt 的激活差中聚类
- 得到多个 reasoning prototype
- 推理时根据当前输入动态生成 steering vector
  参考: [Prototype-Based Dynamic Steering for Large Language Models](https://arxiv.org/abs/2510.05498)

2. `IDS` 和 `PID Steering`  
这两类工作说明：
- steering 强度不该固定，而应该跟输入分布位置动态调整 [In-Distribution Steering: Balancing Control and Coherence in Language Model Generation](https://arxiv.org/abs/2510.13285)
- 干预过程不该只看单层单步，也可以看成跨层反馈控制问题 [Activation Steering with a Feedback Controller](https://arxiv.org/abs/2510.04309)

### 这个 idea 怎么改进

不直接把方法写成“steering paper”，而是把它改成：

`一组 latent reasoning prototype + 动态门控/动态混合`

也就是说：
- 不假设所有题都共享一个 reasoning 向量
- 而是假设有几种不同的“推理原型”
- 每个输入根据自身特征激活不同原型

它可以是：
- test-time guidance
- 训练时的辅助模块
- latent feature selection 的上层结构

### 解决什么问题

主要解决：

1. 固定全局 latent vector 太僵硬  
2. 不同题型可能需要不同 reasoning mode  
3. 一刀切强度容易 oversteer 或 understeer

### 可能取得什么效果

如果做得好，可能看到：

- 比固定向量更稳
- 更适合不同输入
- 在一些复杂推理题上带来更明显提升
- 输出质量下降更少

### 实现难度

**中等偏高。**

比前两个 idea 更复杂，因为需要：
- 原型构造
- 动态选择机制
- 更复杂的验证

### 风险

- 很容易被 reviewer 说成“steering 变体”
- 如果效果不稳定，会像 activation hack，不像核心方法

### 关键参考文献

- [Prototype-Based Dynamic Steering for Large Language Models](https://arxiv.org/abs/2510.05498)
- [In-Distribution Steering: Balancing Control and Coherence in Language Model Generation](https://arxiv.org/abs/2510.13285)
- [Activation Steering with a Feedback Controller](https://arxiv.org/abs/2510.04309)

---

## Idea 4：One-Shot / Small-Data Reasoning Direction Bootstrapping

### 相关论文怎么做

这条线主要参考：

1. `One-shot Optimized Steering Vectors`
- 单样本就能优化出可迁移 steering signal  
  参考: [One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs](https://arxiv.org/abs/2502.18862)

2. `TSV`
- 用少量标注样本起步，再用无标记样本扩展  
  参考: [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)

3. `Feature Extraction and Steering for Enhanced CoT Reasoning`
- 从普通 CoT 或 residual activation 中提取 reasoning direction  
  参考: [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)

### 这个 idea 怎么改进

把这些思想从 steering 转成 latent method 的前处理步骤：

`不是先设计完整大方法，而是先用极少量高质量 reasoning 样本，快速找出可能有用的 latent direction / latent feature。`

然后这些 bootstrap 出来的方向可以用于：
- feature selection
- latent regularization
- orbit consistency
- 动态 guidance 初始原型

也就是说，它本身未必是一篇主论文，但很适合作为一个便宜且有效的启动模块。

### 解决什么问题

主要解决：

1. 你没有大量高质量 latent supervision 数据  
2. 不知道 latent 空间里该往哪个方向找 reasoning signal  
3. 一上来就训大方法成本太高

### 可能取得什么效果

如果做得好，可能带来：

- 极低成本发现有效 latent signal
- 大幅降低方法原型验证成本
- 让后续方法从“完全盲做”变成“先有一个方向”

### 实现难度

**低到中等。**

它很适合先做原型实验。

### 风险

- 小样本方向可能不稳定
- 很可能只能作为辅助模块，而撑不起整篇论文

### 关键参考文献

- [One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs](https://arxiv.org/abs/2502.18862)
- [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)
- [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2505.15634)

---

## Idea 5：Correct-vs-Wrong Latent Separation for Reasoning

### 相关论文怎么做

这条线主要受两类工作启发：

1. `TSV`  
目标不是直接提分，而是让 truthful 和 hallucinated 表示更可分  
参考: [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)

2. 一系列 latent reasoning 机制论文  
它们说明：
- 正确推理和错误推理在 latent space 里未必完全混在一起
- 错误路径、shortcut 路径、稳定路径之间可能存在结构差异  
这些判断主要受 [Do Latent Tokens Think?](https://arxiv.org/abs/2512.21711)、[Dynamics Within Latent Chain-of-Thought](https://arxiv.org/abs/2602.08783)、[Do Latent-CoT Models Think Step-by-Step?](https://arxiv.org/abs/2602.00449) 启发

### 这个 idea 怎么改进

把目标从“直接提高准确率”改成：

`先让正确 reasoning 和错误 reasoning 在 latent feature 空间里更容易分开。`

然后基于这种分离，再去做：
- regularization
- reranking
- latent guidance

所以这是一个“先判别，再增强”的路线。

### 解决什么问题

主要解决：

1. 直接做 latent reasoning enhancement 太黑箱  
2. 不知道哪些 latent signal 是好信号，哪些是坏信号  
3. 错误轨迹可能和正确轨迹混在一起，方法难以稳定

### 可能取得什么效果

如果做得好，可能带来：

- 更容易筛 feature
- 更容易做错误检测
- 让后续 latent regularization 更有目标
- 有可能带来更稳的 reasoning enhancement

### 实现难度

**中等。**

需要有正确 / 错误轨迹样本，但不一定需要特别大模型。

### 风险

- 分得开不等于能提分
- reviewer 可能会说这是分析工具，不是完整方法

### 关键参考文献

- [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917)
- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)

---

## Idea 6：Causal-Necessity Regularization for Latent Steps

### 相关论文怎么做

这条线主要来自：

1. `Dynamics Within Latent Chain-of-Thought`
2. `Do Latent Tokens Think?`
3. `Do Latent-CoT Models Think Step-by-Step?`

对应参考：
- [Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)
- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)

这些论文都在问：
- latent step 是否真的有因果作用
- 干预 latent step 后，输出会不会系统性变化
- 某些 latent step 会不会只是“看起来像推理”，实际上并不必要

### 这个 idea 怎么改进

从分析走向训练目标：

`如果某些 latent step 真的是关键推理步骤，那么扰动它们时，模型应该明显受影响；如果怎么扰都不影响，说明这些步骤可能不是真正必要的。`

据此可以在训练里加一个更弱的因果必要性约束：
- 不要求严格因果证明
- 但惩罚那些“看起来有很多 latent step，实际上并不依赖这些 step”的轨迹

### 解决什么问题

主要解决：

1. latent reasoning 可能只是形式上有很多 step  
2. 很多 step 可能只是冗余或假推理  
3. 最终方法可能提分，但机制不可信

### 可能取得什么效果

如果做得好，可能带来：

- 减少假推理步骤
- 提高 latent trajectory 的有效性
- 让 latent reasoning 更像真实中间计算

### 实现难度

**偏高。**

比前几个 idea 风险都大。

### 风险

- reviewer 会非常敏感“causal”这个词
- 很容易被质疑你只是做了扰动，不是真的识别了因果必要性

### 关键参考文献

- [Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](https://arxiv.org/abs/2602.08783)
- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks](https://arxiv.org/abs/2602.00449)

---

## 最后给一个现实排序

如果按“最像现在能真正启动的研究 idea”排序，我会建议：

1. **Idea 1: Feature-Selected Rationale-Orbit Regularization**
2. **Idea 2: Latent Feature Regularization**
3. **Idea 5: Correct-vs-Wrong Latent Separation**
4. **Idea 3: Prototype-Based Dynamic Latent Guidance**
5. **Idea 4: One-Shot / Small-Data Reasoning Direction Bootstrapping**
6. **Idea 6: Causal-Necessity Regularization**

## 一句话区分它们

- `Idea 1`：最完整，最像主论文
- `Idea 2`：最稳，最容易起步
- `Idea 3`：最像“高级版 latent steering”
- `Idea 4`：最适合做低成本启动模块
- `Idea 5`：最适合做“先判别再增强”
- `Idea 6`：最有研究味，但也最容易被质疑
