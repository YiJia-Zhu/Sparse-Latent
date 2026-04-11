# Reframed Review: Negative-Controlled Selective Invariance

Date: 2026-04-11

## 1. Minimum Experiment Package Ranked by Acceptance Lift per GPU Week

### Rank 1: Main behavioral result on one dataset
- Dataset: `GSM8K`
- Model: one credible small backbone only, preferably `Llama-3.2-1B-Instruct`-scale latent reasoning setup
- Conditions:
  - augmentation-matched no-regularization
  - full-state orbit regularization
  - selected-mask without negative control
  - selected-mask with negative control
- Seeds: `2`
- Primary metric: accuracy-conditioned held-out rewrite consistency
- Secondary: task accuracy, consistency on correct-only subset, consistency on all samples
- Critical protocol:
  - train rewrites and eval rewrites must come from different prompts / generation settings
  - keep `k`, `lambda`, train budget matched
- Why this is rank 1:
  - this is the only experiment that directly tests the paper’s main claim
  - if this fails, the project is not publishable

### Rank 2: Sparse-mask control to isolate negative control from sparsity
- Run on the same main setup as Rank 1
- Add:
  - random top-k mask
  - orbit-stability-only top-k mask
- Seeds: `1-2`
- Goal:
  - show the gain is not just “sparse regularization beats full-state”
- Why this is rank 2:
  - reviewers will otherwise say the method wins because it regularizes less

### Rank 3: Cheap causal necessity check on selected coordinates
- No retraining required after Rank 1
- Compare:
  - selected-neg top-k
  - selected-no-neg top-k
  - random top-k
- Intervention:
  - zero selected coordinates at evaluation time
- Metrics:
  - answer logit drop
  - correct-answer rate drop
  - rewrite-consistency drop
- Why this is rank 3:
  - this is the cheapest evidence that the chosen coordinates matter behaviorally
  - it supports the “less answer-generic, still useful” claim without invoking faithfulness

### Rank 4: Small replication
- Prefer a second dataset over a second tiny model
- Best option if feasible: `ARC-Challenge` or another non-arithmetic reasoning set with generated rationales
- Minimal conditions:
  - augmentation-matched no-reg
  - selected-no-neg
  - selected-neg
- Seeds: `1`
- Why this is rank 4:
  - same-answer negative control on arithmetic can look dataset-specific
  - one non-arithmetic replication materially lowers reviewer skepticism

### Rank 5: Analysis-only selector sanity checks
- Keep the current non-training pilot as appendix / motivation only
- Do not make it a main result
- Why this is rank 5:
  - useful as a feasibility check
  - weak as publishable evidence

## 2. Minimal Decisive Package

If compute is tight, the minimum credible package is:

1. `GSM8K` main experiment with four conditions and `2` seeds
2. One sparse-mask control on the same setup
3. One post-training causal ablation study
4. One tiny replication with three conditions and `1` seed

If Rank 1 does not show a clear win for `selected-neg` over both `full-state` and `selected-no-neg`, stop.

## 3. Results-to-Claims Matrix

| Outcome pattern | Defensible claim | Must drop |
|---|---|---|
| `selected-neg` clearly beats `full-state` and `selected-no-neg` on accuracy-conditioned held-out rewrite consistency, with similar accuracy; small replication agrees | same-answer negative control improves selective invariance regularization for rewrite robustness; full-state alignment is not the best way to impose rewrite consistency | any claim about faithful reasoning or identifying abstract reasoning coordinates |
| `selected-neg` beats `full-state`, but ties `selected-no-neg` | selective sparse invariance can help rewrite robustness; full-state alignment may be over-broad | claim that negative control is the key ingredient |
| `selected-neg` beats baseline slightly, but not significantly and not above `selected-no-neg` | method is a weak heuristic worth further study | top-tier novelty / robust method claim |
| `selected-neg` improves raw consistency but not accuracy-conditioned consistency | model becomes more behaviorally consistent under rewrites, possibly by stabilizing errors | robustness claim in the paper’s title or main contribution |
| `selected-neg` improves consistency but hurts task accuracy materially | there is a robustness-accuracy tradeoff under selective invariance | claim that the method is a practical training improvement |
| non-training selector diagnostics look good, but training gives no behavioral gain | selector changes representation statistics in the intended direction | any claim that the selector improves training outcomes |
| causal ablation shows selected-neg coordinates are not more behaviorally important than random / selected-no-neg | negative control may change the mask, but causal relevance remains unproven | “output-relevant selected coordinates” claim |
| main dataset positive, replication fails badly | dataset-specific promise on the main benchmark | any general method claim |

## 4. Mock Top-Tier Review: Positive Case

### Summary
The paper studies rewrite-consistency regularization for latent reasoning models and argues that consistency should be imposed only on a selected subset of latent coordinates. The main technical idea is a same-answer negative control used during coordinate selection to filter answer-generic features before applying a sparse invariance loss. On a 1B-scale latent reasoning setup, the proposed method improves accuracy-conditioned held-out rewrite consistency over augmentation-only, full-state regularization, and selection without negative control, while preserving task accuracy.

### Strengths
- Clear, compact intervention with a testable thesis
- Stronger framing than many “faithful reasoning” papers because it avoids overclaiming
- The key comparison against `selected-no-neg` isolates the real novelty
- Accuracy-conditioned evaluation is the right primary metric
- The paper includes at least one causal coordinate-ablation study instead of relying only on representation diagnostics

### Weaknesses
- Empirical scope is still narrow for a top-tier venue
- Coordinate-level interpretation remains basis-dependent
- The method addresses answer-cue contamination, but not other shortcut channels
- The rewrite generation pipeline is still a potential confound

### Questions for Authors
- How sensitive are results to the answer-bucketing scheme used for the negative control?
- Does the method still help under a different rewrite source or teacher?
- How much of the gain remains if the mask is recomputed during training?

### Score
- `6/10` borderline accept

### Confidence
- `4/5`

### What Would Move Toward Accept
- one more convincing replication
- stronger evidence that gains are not an artifact of the rewrite generator

## 5. Mock Top-Tier Review: Weak Case

### Summary
The paper proposes negative-controlled selection of latent coordinates for rewrite-consistency regularization. While the idea is sensible, the empirical evidence does not convincingly establish that the negative control itself matters: the proposed method only marginally improves over baseline and does not clearly outperform selected-mask regularization without negative control.

### Strengths
- The method is simple
- The paper asks a legitimate question about where consistency should be imposed

### Weaknesses
- The core novelty is not empirically validated
- Improvements are small and could easily come from noise or generic sparsity effects
- The analysis diagnostics are too aligned with the selection rule to count as strong evidence
- Experimental scope is too limited for a top-tier venue

### Questions for Authors
- Why should the reader believe same-answer negative control is necessary if `selected-no-neg` performs similarly?
- Can the authors show robustness under a genuinely different rewrite source?
- Are the reported gains statistically reliable?

### Score
- `3/10` reject

### Confidence
- `4/5`

### What Would Move Toward Accept
- a clear behavioral win over `selected-no-neg`
- a stronger sparse-mask control analysis

## 6. Safe Title and Abstract Framing

### Title
Negative-Controlled Selective Invariance for Rewrite Robustness in Latent Reasoning Training

### Abstract skeleton
We study how to impose rewrite-consistency regularization during latent reasoning training without aligning the entire latent state. Our method selects a fixed sparse mask of latent coordinates using three offline signals: stability across same-question rationale rewrites, output relevance, and a same-answer negative control that filters answer-generic features. We then regularize only the selected coordinates during LoRA fine-tuning. On small latent reasoning backbones, this negative-controlled selective invariance improves accuracy-conditioned held-out rewrite consistency over augmentation-only training, full-state alignment, and selection without negative control, while maintaining comparable task accuracy. We present this as a robustness method, not as evidence that the selected coordinates uniquely represent true reasoning steps.

## 7. One Current Element to Kill

Kill `OrbitMinusSameAnswer` as a headline result.

Reason:
- it is too close to the selector’s own objective
- reviewers will read it as circular evidence
- keep it only as an internal sanity check or appendix diagnostic

If one storyline element also needs to die, kill all language suggesting discovery of stable “reasoning neurons” or stable coordinate identities.
