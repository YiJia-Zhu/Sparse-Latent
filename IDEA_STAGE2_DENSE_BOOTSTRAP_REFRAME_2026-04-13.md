# Stage-2 Reframe: Dense Bootstrap then Selective Refinement

Date: 2026-04-13

## 1. What failed in the previous round

The April 13 official-match runs gave:

- `plain Llama-3.2-1B-Instruct`: `0.3616` GSM8K test accuracy
- `full_state_official_match_20260413_000931`: `0.4193`
- `sparse_neg_official_match_20260413_000931`: `0.3707`
- `sparse_no_neg_official_match_20260413_000931`: `0.0349`

This is not clean evidence that negative control is useless. It is stronger evidence that the previous comparison conflated two problems:

1. learning a latent-capable CODI-style scaffold from a plain LLaMA checkpoint
2. deciding which latent coordinates should receive selective distillation

`full_state` can solve both at once because it receives dense teacher signal. `sparse_*` receives only a tiny coordinate subset while still being asked to bootstrap latent reasoning from scratch.

## 2. Re-survey: what the recent literature changes in our interpretation

### A. Progressive / hybrid latent training is now the more defensible default

- [Hybrid Reasoning Policy Optimization (HRPO), arXiv:2505.18454](https://arxiv.org/abs/2505.18454)
  argues that latent reasoning should be integrated progressively rather than treated as a brittle all-or-nothing substitution.
- [System-1.5 Reasoning, arXiv:2505.18962](https://arxiv.org/abs/2505.18962)
  explicitly mixes latent and language-space shortcuts instead of assuming every reasoning step should stay in one space.

Takeaway:

> A staged or hybrid route is more plausible than forcing sparse latent supervision to do cold-start backbone formation by itself.

### B. Feature-level interventions can help consistency, but hard suppression is risky

- [LF-Steering, arXiv:2501.11036](https://arxiv.org/abs/2501.11036)
  shows that feature-level control helps semantic consistency because whole-state interventions are too entangled.
- [Measuring and Mitigating Post-hoc Rationalization in Reverse CoT Generation, arXiv:2602.14469](https://arxiv.org/abs/2602.14469)
  shows that naive suppression of answer-conditioned signals can backfire; structure-aware redirection works better than blunt forbidding.

Takeaway:

> Our old `selected-neg` intuition is still reasonable, but a hard sparse replacement may be too aggressive. A refinement-style intervention is safer than a from-scratch sparse-only intervention.

### C. Confidence / routing failures are a real issue in latent reasoning

- [ThinkRouter, arXiv:2602.11683](https://arxiv.org/abs/2602.11683)
  reports that low-quality latent trajectories can accumulate noise and overconfidence, motivating confidence-aware routing between latent and discrete reasoning.

Takeaway:

> If sparse supervision is weak early, the model may enter exactly the noisy latent regime that later papers warn about. This again favors dense bootstrap first.

## 3. Revised idea

The new thesis is no longer:

> “Use `selected-neg` instead of `full_state` from the first training step.”

It becomes:

> “Use dense full-state distillation to bootstrap a workable latent scaffold, then apply negative-controlled selective refinement to clean answer-generic coordinates without destroying the scaffold.”

In short:

- Stage 1: `full_state` builds the latent reasoning backbone
- Stage 2: `selected_no_neg` / `selected_neg` act as refinement objectives, not cold-start objectives

This is a smaller and more defensible change than inventing a new latent backbone.

## 4. Experiment plan for this reframe

Using the April 13 `full_state` checkpoint as the shared stage-1 initialization, run:

1. `full_state continue`
2. `sparse_no_neg refine from full_state`
3. `sparse_neg refine from full_state`

Decision rule:

- If `sparse_neg refine` beats `full_state continue`, the old failure was mostly a training schedule mistake.
- If `sparse_neg refine` beats `sparse_no_neg refine` but not `full_state continue`, then negative control still helps but the current refinement strength is insufficient.
- If both sparse refinements fail badly even from `full_state`, then the current selector should be softened or replaced.

## 5. Runs launched on 2026-04-13

The following stage-2 runs were launched locally:

- `full_state_refine_from_full_20260413_a`
- `sparse_neg_refine_from_full_20260413_a`
- `sparse_no_neg_refine_from_full_20260413_a`

Launcher added:

- `codi_local_sparse/run_stage2_refine_official_match.sh`

This file exists so the stage-2 bootstrap-refinement setting is reproducible and can be rerun with either full data or a smaller pilot budget.

## 6. Quick pilot results

To avoid waiting hours for the first signal, a smaller pilot was run with:

- `exp_mode=True`
- `exp_data_num=256`
- `eval_max_samples=200`
- single-GPU runs from the same `full_state` warm start

Results:

| Run | Accuracy on 200 eval examples |
|---|---:|
| `full_state_refine_from_full_pilot_20260413_b` | `0.350` |
| `sparse_no_neg_refine_from_full_pilot_20260413_b` | `0.345` |
| `sparse_neg_refine_from_full_pilot_20260413_b` | `0.360` |

Reading:

- the revised stage-2 idea is directionally supported
- `selected-neg` now beats both `selected-no-neg` and `full_state` under the same warm-start scaffold
- this strongly suggests the old failure was mostly a cold-start training-design issue, not a clean rejection of negative control

The full-data long run being kept alive is:

- `sparse_neg_refine_from_full_20260413_a`

## 7. Low-LR follow-up pilots changed the ranking

The first stage-2 pilots above still used the original cold-start learning rate
(`8e-4`), which is too aggressive for refinement from a strong `full_state`
checkpoint. A follow-up pilot repeated the same stage-2 setup with:

- same warm start from `full_state_official_match_20260413_000931`
- same small train budget (`exp_mode=True`, `exp_data_num=256`)
- same evaluation slice (`200` test examples)
- lower refinement LR: `1e-4`

Results:

| Run | Accuracy on 200 eval examples |
|---|---:|
| `full_state_refine_from_full_pilot_lr1e4_20260413_c` | `0.430` |
| `sparse_no_neg_refine_from_full_pilot_lr1e4_20260413_c` | `0.440` |
| `sparse_neg_refine_from_full_pilot_lr1e4_20260413_c` | `0.435` |

Comparison to prior references on the same 200-example slice:

| Reference | Accuracy |
|---|---:|
| old `full_state_official_match_20260413_000931` | `0.440` |
| old cold-start `sparse_neg_official_match_20260413_000931` | `0.385` |

Updated reading:

- the key fix was not merely "use `selected_neg`", but "use dense bootstrap plus
  a much smaller refinement LR"
- once the LR is corrected, all warm-start stage-2 variants recover strongly
- `selected_no_neg` currently ties the old `full_state` baseline on the pilot
  slice, while `selected_neg` is slightly below but still much better than the
  old cold-start sparse run
- this means the previous high-LR ranking (`selected_neg > full_state >
  selected_no_neg`) was not stable enough to use as the final decision rule

## 8. Current execution decision

Because the low-LR pilot is the first setting that nearly reaches or matches the
old baseline, the next correct move is a fair full-data rerun of all three
stage-2 refinements under the same low-LR schedule:

- `full_state_refine_from_full_lr1e4_20260413_d`
- `sparse_no_neg_refine_from_full_lr1e4_20260413_d`
- `sparse_neg_refine_from_full_lr1e4_20260413_d`

These runs were launched on 2026-04-13 using the local idle GPUs after stopping
the obsolete high-LR long run.
