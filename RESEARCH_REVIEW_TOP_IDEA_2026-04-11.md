# Top Idea Research Review

日期：2026-04-11  
主题：`Negative-Controlled Orbit / Selective Invariance`  
外部 reviewer agent：`019d7831-a57d-7e42-bfb5-012013a17205`  
reviewer model：`gpt-5.4`  
reasoning effort：`xhigh`

## 1. 审查对象

当前 top idea 的原始叙事是：

- 在纯 LLM latent reasoning 中，现有方法虽然可能提分，但很难说明 latent step 学到的到底是抽象推理结构，还是答案线索、模板压缩和伪推理
- 我们想做的核心方法是：
  - 先在 warm-start checkpoint 上，用 `OrbitStability + OutputNecessity - SameAnswer contamination filtering` 选出一组固定 top-k 坐标
  - 再只对这组坐标做 rewrite consistency / orbit invariance regularization
- 原始较强 claim 是：
  - 这些坐标更接近“faithful latent reasoning”

当前已有证据：

- 已完成一批 analysis-only 非训练验证
- `selected_neg` 在 11/11 个 run 里比 `selected_no_neg` 有更高的 `OrbitMinusSameAnswer`
- 但尚无正式训练结果

## 2. Reviewer 总结结论

reviewer 的判断很直接：

> 现在这个版本还不是 top-venue 论文。  
> 它更像一个 plausible heuristic 加上一组还不错的 sanity check，claim 明显走在证据前面。

更具体地说：

- 作为“faithful latent reasoning”论文：
  - 不成立，当前大概率被拒
- 作为“negative-controlled selective invariance improves rewrite robustness”方法论文：
  - 有可能成立
  - 但前提是训练结果必须干净，而且控制组必须更强

reviewer 明确建议：

> 立刻删掉 `faithful` 这个主叙事，改讲  
> `confound-reduced rewrite robustness under latent reasoning training`

## 3. 第一轮主要批评

### 3.1 最大逻辑漏洞

1. `rationale rewrite invariance != faithful reasoning`

- 同一题目不同 rationale 下表示更稳定，不代表它学到了抽象推理结构
- 它也可能只是抓住了：
  - 题目 identity
  - 题型
  - 运算模式
  - lexical overlap
  - difficulty pattern

2. `same-answer negative control` 只去掉了一个 confound

- 它只能说明你在努力避免答案线索污染
- 不能说明剩下的东西就是真正推理

3. `OutputNecessity` 不是 faithfulness 证据

- 一个坐标影响正确答案 token，不等于它是 reasoning coordinate
- 它也可以是 shortcut、answer prior、模板 cue

4. 坐标解释是 basis-dependent 的

- hidden states 的坐标没有天然唯一意义
- 如果 selected coordinate id 在不同 run 下不稳定，那么“我找到了 reasoning 坐标”的解释非常脆弱

5. 固定 warm-start mask 可能在训练后失效

- 训练后模型可以把信息迁移到别的坐标或别的层
- 那么你 regularize 的这批坐标看似更干净，但模型照样可能绕开它作弊

6. `OrbitMinusSameAnswer` 太接近 selector 自身

- 这是一个 sanity check
- 不是论文主证据
- reviewer 会觉得有循环论证嫌疑

7. `orbit` 这个说法可能过度包装

- 如果没有更正式的群作用或等价结构定义
- 有些 reviewer 会把它看成 pairwise rewrite consistency 的修辞包装

### 3.2 当前非训练证据的评价

reviewer 认为：

- 不是没用
- 但对顶会标准来说，主要还是 feasibility check

它能说明：

- selector 不是完全退化的
- negative control 确实改变了选中的 channel family
- 这种差异在多个近邻设定下可复现

它不能说明：

- 这些 channel 编码 reasoning
- regularizing 它们真的改善训练行为
- full-state alignment 失败的原因就是你说的那个机制

一句话概括：

> 当前非训练结果可以作为 go / no-go filter，不能作为 paper 主证据。

## 4. Reviewer 建议的重构 framing

reviewer 认可的更安全表述是：

### 可接受 framing

- 不是：
  - `faithful latent reasoning`
- 而是：
  - `negative-controlled selective invariance for rewrite robustness in latent reasoning training`

### 可接受主命题

不是说：

- “我们找到了真正推理坐标”

而是说：

- “在 latent reasoning training 中，如果要施加 rewrite consistency regularization，不应该对全状态一刀切，而应该只对一组经过 same-answer negative control 过滤后的坐标做 selective invariance；这比 full-state 或 no-negative-control 的版本更能提升 held-out rewrite robustness，同时保持相近 accuracy。”

### reviewer 给出的安全标题

`Negative-Controlled Selective Invariance for Rewrite Robustness in Latent Reasoning Training`

## 5. 最小可救实验包

reviewer 给出的最高性价比实验排序如下。

### Rank 1

主行为结果，必须做：

- 数据集：`GSM8K`
- backbone：一个可信的小 backbone，优先 `Llama-3.2-1B-Instruct` 级别
- 条件：
  - `augmentation-matched no-reg`
  - `full-state orbit regularization`
  - `selected-no-neg`
  - `selected-neg`
- seeds：至少 `2`
- 主指标：
  - `accuracy-conditioned held-out rewrite consistency`
- 次指标：
  - task accuracy

reviewer 直说：

> 这张表就是论文本体。  
> 如果 `selected-neg` 不能明显优于 `selected-no-neg` 和 `full-state`，项目就该停。

### Rank 2

稀疏控制组：

- `random top-k mask`
- `OrbitStability-only top-k`

目的：

- 反驳 reviewer 最容易提出的攻击：
  - “你只是 regularize 得更少，所以看起来更好”

### Rank 3

训练后因果干预分析：

- 对 `selected-neg`、`selected-no-neg`、`random` 三组 mask 做 zeroing
- 测：
  - answer logit drop
  - correct-answer rate drop
  - rewrite-consistency drop

reviewer 判断：

> 这是最便宜、但最能把“坐标不是装饰品”讲清楚的实验。

### Rank 4

一个小 replication：

- reviewer 更希望是第二个数据集，而不是再加一个很弱的小模型
- 最好不是纯 arithmetic
- 优先级上 `ARC` 高于再做一个数学集

### Rank 5

你现在做的 analysis-only pilot：

- 可以保留
- 但应该进 appendix 或 motivation
- 不要当主结果

## 6. Results-to-Claims Matrix

| 结果模式 | 还能保留的 claim | 必须删掉的 claim |
|---|---|---|
| `selected-neg` 明显优于 `full-state` 和 `selected-no-neg`，accuracy 接近，且 replication 方向一致 | same-answer negative control improves selective invariance regularization for rewrite robustness | faithful reasoning、abstract reasoning coordinates |
| `selected-neg` 优于 `full-state`，但和 `selected-no-neg` 打平 | sparse selective invariance 可能优于 full-state | negative control 是核心 novelty |
| `selected-neg` 只比 baseline 略好，且不显著优于 `selected-no-neg` | 一个弱 heuristic，可以继续研究 | top-tier method claim |
| raw consistency 提升，但 accuracy-conditioned consistency 不提升 | 输出更一致了，但可能只是更稳定地犯错 | robustness 主张 |
| consistency 提升但 accuracy 掉很多 | 存在 robustness-accuracy tradeoff | practical training improvement |
| analysis-only diagnostics 很漂亮，但训练行为没有提升 | selector 改变了表示统计 | selector improves training outcomes |
| causal ablation 不比 random / no-neg 更强 | mask 变了，但行为相关性没证实 | selected coordinates are behaviorally special |
| 主数据集有效，但 replication 明显失败 | 在该 benchmark 上有局部效果 | general method claim |

## 7. 模拟评审结论

### 如果主结果是正的

即：

- `selected-neg` 明显优于 `selected-no-neg` 和 `full-state`
- accuracy 差不多
- 至少一个小 replication 方向一致

reviewer 给出的预估是：

- 分数：`6/10`
- 置信度：`4/5`
- 级别：borderline accept

也就是说：

- 这不是“稳收”论文
- 但在叙事收紧、实验做干净后，有进入讨论区的可能

### 如果结果偏弱

即：

- 只略优于 baseline
- 或者不优于 `selected-no-neg`

reviewer 给出的预估是：

- 分数：`3/10`
- 置信度：`4/5`

这基本意味着：

- 核心 novelty 没被验证
- 论文会塌成一个“带一些漂亮内部诊断的 sparse regularization 故事”

## 8. 当前最该砍掉的东西

reviewer 点名建议删掉：

### 第一优先

删掉 `faithful`

- 当前证据完全不支撑
- 会强烈触发 reviewer 的反感

### 第二优先

不要把 `OrbitMinusSameAnswer` 当 headline result

- 这个量太贴近 selector 自己
- 更适合放 appendix

### 第三优先

不要讲“稳定 reasoning neurons / 真推理坐标”

- 当前坐标 id 本身跨 run 不稳定
- 稳定的是 selection criterion，不是某一组唯一 neuron

### 第四优先

`gpt2` 不适合进主文

- 可以留在 feasibility check
- 放主文会���低说服力

## 9. 当前 reviewer 共识下的最合理推进路线

最合理路线不是继续堆 non-training 分析，也不是立刻大规模 benchmark。

而是：

1. 彻底改 framing
2. 只做最小训练主表
3. 如果主表不赢，就尽快停
4. 如果主表赢，再补稀疏控制、因果 zeroing 和一个小 replication

可执行版本如下：

1. 主表：
   - `aug-no-reg`
   - `full-state`
   - `selected-no-neg`
   - `selected-neg`
   - `2 seeds`
2. 加一个 `random top-k`
3. 做 post-training zeroing
4. 做一个小 replication

## 10. 我们现在该怎么理解这个项目

reviewer 给出的最终定位其实很明确：

> 这不是一个“faithfulness / mechanism discovery”项目。  
> 如果能成，它更像一个小而精的训练方法论文：
> `same-answer negative control helps sparse latent consistency regularization improve rewrite robustness`

这个定位下：

- 论文会窄很多
- 但更真实
- 也更容易被实验支撑

## 11. 附件

reviewer 第二轮整理稿：

- [REVIEW_REFRAMED_MIN_PACKAGE.md](./REVIEW_REFRAMED_MIN_PACKAGE.md)
