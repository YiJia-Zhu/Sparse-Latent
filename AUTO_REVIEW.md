# Auto Review

日期：2026-04-11  
模式：`internal reviewer-style stop review`

## 1. 审查对象

本轮审查的不是原始主线本身，而是新补充的 extension：

`Confidence-Gated Selective Invariance`

目标是回答：

- 它能不能缓解第二轮里出现的 accuracy / robustness tradeoff
- 同时保住 `selected-neg > selected-no-neg` 的弱优势

## 2. 审查结论

结论：`NO-GO AS MAIN EXTENSION`

原因：

1. 它确实恢复了一部分平均准确率。
2. 但核心主指标 `accuracy-conditioned rewrite consistency` 下降。
3. `selected-neg-gated` 没有再优于 `selected-no-neg-gated`。

因此，这个 extension 不能支持更强的论文 claim。

## 3. Results-to-Claims

当前还能保留的说法：

- 按样本不确定性调节 orbit regularization 是一个可运行的训练扩展
- 它可能缓解一部分 accuracy drop

当前不能保留的说法：

- gated 能加强 negative control 的收益
- gated 是当前主线的更优版本
- gated 提高了 rewrite robustness

## 4. 决策

- 不进入多轮 auto-review-loop
- 不把 gated extension 升级成新主线
- 保留代码和结果，后续若继续，只做极小超参 sweep

## 5. 下一步如果还要继续

只建议做低成本工作：

1. `selected-neg` / `selected-no-neg` 多 seed
2. 轻扫 `lambda_orbit`
3. 更独立的 held-out rewrite
4. 若主线重新变强，再补 `ARC` replication
