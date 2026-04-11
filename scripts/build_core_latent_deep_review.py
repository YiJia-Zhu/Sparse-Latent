#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "latent_llm_papers.json"
OUT = ROOT / "CORE_LATENT_LLM_DEEP_REVIEW.md"


GROUPS = [
    (
        "Foundations",
        "这些论文定义了 latent reasoning 的基本操作方式：压缩 CoT、把推理搬进隐空间、或直接从激活中提取 reasoning signal。",
        [
            "2409.08561",
            "2409.14026",
            "2410.13640",
            "2411.04282",
            "2412.06769",
            "2412.13171",
            "2412.17747",
            "2502.03275",
        ],
    ),
    (
        "Training And Distillation",
        "这一组关心怎样把显式推理蒸馏成 latent thought，或怎样直接训练 continuous/implicit reasoning。",
        [
            "2502.12134",
            "2502.21074",
            "2503.18866",
            "2505.15778",
            "2506.11752",
            "2509.20317",
            "2602.08220",
            "2602.08332",
        ],
    ),
    (
        "Test-Time Compute, Loops, And Routing",
        "这一组是目前最拥挤的主线：通过 recurrent depth、loop、adaptive stopping、hybrid routing 等方式把 test-time compute 做到 latent space 里。",
        [
            "2502.05171",
            "2502.17416",
            "2505.11484",
            "2505.18454",
            "2505.18962",
            "2506.18582",
            "2510.25741",
            "2511.21581",
            "2602.10520",
            "2602.11451",
            "2602.11683",
            "2602.14759",
            "2603.01914",
            "2603.04948",
        ],
    ),
    (
        "Diagnostics, Mechanism, And Limits",
        "这一组最重要，因为它开始追问 latent token 到底是不是在做真实计算，以及什么时候会出现 silent failure、shortcut compression 或 planning failure。",
        [
            "2504.10615",
            "2509.25239",
            "2512.21711",
            "2601.08058",
            "2602.00449",
            "2602.01148",
            "2602.08783",
            "2603.03475",
            "2603.06222",
        ],
    ),
]


NOTES = {
    "2409.08561": ("把显式 CoT 压缩成 special latent representation，是早期“silent thought/hidden CoT”路线的起点之一。", "中，适合当 baseline，不适合作为今天的新主线。"),
    "2409.14026": ("证明激活空间里确实可以注入 reasoning vector，这为“latent feature steering”提供了可操作证据。", "中，适合启发诊断工具，不适合直接当主方法。"),
    "2410.13640": ("把 latent trajectory 用于 output-free self-evaluation，而不是直接提升答案质量，这个切口很聪明。", "中，适合做辅助判别器，不是你当前最优主线。"),
    "2411.04282": ("LaTRO 是较早把 latent reasoning 明确写成可优化分布的工作，方法上比较完整。", "中，值得了解，但训练思路现在已不算稀缺。"),
    "2412.06769": ("Coconut 是领域标志论文，几乎定义了“continuous chain of thought”这条主线。", "高，必须精读，是很多后续工作的共同祖先。"),
    "2412.13171": ("Compressed CoT 说明 latent compression 不只是省 token，也是一种新的 reasoning carrier。", "中，更多像效率线代表作。"),
    "2412.17747": ("用 differentiable cache 做 latent deliberation，说明 latent reasoning 可以不靠显式 token 链。", "中，偏机制设计，复现价值高于立项价值。"),
    "2502.03275": ("把 latent token 和 text token 混用，代表 hybrid reasoning 的早期版本。", "中，适合作为你后面 hybrid baseline。"),
    "2502.12134": ("SoftCoT 的重要性在于“训练-free continuous reasoning”，大幅降低实验门槛。", "高，非常适合你当前算力条件下做 comparative study。"),
    "2502.21074": ("CODI 把 CoT 压进 continuous space via self-distillation，是 latent distillation 线的代表。", "高，你后续若做 invariance/faithfulness，CODI 是必须对照的目标。"),
    "2503.18866": ("把 latent thoughts 和 data efficiency/pretraining 绑定，视角比单纯推理更大。", "中，观点重要，但离你当前小算力设置稍远。"),
    "2505.15778": ("Soft Thinking 继续扩展 continuous concept space 的训练-free/low-cost 视角。", "中，适合作为“连续概念空间”类工作代表。"),
    "2506.11752": ("DART 用自蒸馏把 autoregressive CoT 迁移到 silent thought，是很自然也很强的工程路线。", "高，适合做 teacher-student consistency 相关对照。"),
    "2509.20317": ("SIM-CoT 的关键在 step-level supervision，试图填平 implicit CoT 和 explicit CoT 的性能鸿沟。", "高，和你若做 supervised latent states 很相关。"),
    "2602.08220": ("把 latent CoT 做到 token-level adaptive pretraining，回答的是“每个 token 前是否需要不同思考预算”。", "中，思想重要，但预训练成本偏高。"),
    "2602.08332": ("Thinking States 很值得注意，因为它把 reasoning 放到 input processing 期间而非输出前。", "高，这类时间布局变化很可能启发新 benchmark。"),
    "2502.05171": ("Recurrent Depth 明确把 latent reasoning 和 inference-time scaling 绑定，是 loop/ponder 系列基石之一。", "高，必须精读，是后面很多 loop paper 的参照系。"),
    "2502.17416": ("Looped Transformers 提出“问题需要更多 depth 而不是更多参数”，这是整个 latent loop 路线的核心命题。", "高，你的 phase-diagram 方向必须覆盖它。"),
    "2505.11484": ("SoftCoT++ 把 latent reasoning 接到 test-time scaling 上，说明 continuous thought 也能做多路径探索。", "中，适合做 test-time exploration baseline。"),
    "2505.18454": ("Hybrid Latent Reasoning via RL 表明 latent/text 切换已经从 heuristic 走向 RL-based control。", "中，说明 routing 赛道已经很拥挤。"),
    "2505.18962": ("System-1.5 代表 latent-explicit 混合计算的思路，追求快与准之间的动态折中。", "中，适合引用说明混合式 reasoning 已经不是空白。"),
    "2506.18582": ("PCCoT 很重要，因为它攻击了 continuous CoT 的顺序依赖问题，提出并行更新。", "中，偏效率和训练技巧，对你主线帮助有限。"),
    "2510.25741": ("Ouro/LoopLM 把 latent reasoning 前移到 pretraining，规模也更大，是 looped LM 的强化版。", "高，必须读，因为它把 loop 路线推到了更完整的系统层面。"),
    "2511.21581": ("Adaptive stopping 是 latent reasoning 里很自然的问题，这篇把“何时停止”变成 RL 目标。", "中，适合作为 routing/compute allocation 的代表引用。"),
    "2602.10520": ("RLTT 说明 final-state reward 不适合 looped latent computation，必须奖励 trajectory。", "高，这对你后面若做 process-based objective 很关键。"),
    "2602.11451": ("LoopFormer 关注 elastic-depth，强调预算可控而不是固定循环数。", "中，适合作为 budget-conditioned baseline。"),
    "2602.11683": ("ThinkRouter 说明 latent vs discrete 切换已经被系统研究，单独做 router 很难新。", "高，主要价值是帮你排除重复方向。"),
    "2602.14759": ("Inner Loop Inference 很有价值，因为它主张无需训练即可释放 latent iterative computation。", "高，和你小算力条件高度匹配。"),
    "2603.01914": ("AdaPonderLM 把 token-wise adaptive depth 做得更细，说明 pondering 已进入精细化阶段。", "中，适合 phase diagram 中当 adaptive-depth 代表。"),
    "2603.04948": ("∇-Reasoner 把 latent refinement 做成 test-time gradient descent，是与 sampling/loop 不同的一条线。", "中，概念新，但实现和稳定性要求更高。"),
    "2504.10615": ("这是 benchmark 线的重要起点，开始正面衡量 latent-space reasoning 能力而非只报 accuracy gain。", "高，这是你最该沿着继续挖的方向。"),
    "2509.25239": ("Formal Comparison 提供理论层面的 CoT vs latent thought 区别，适合做 framing。", "中，理论上有用，但不直接给实验方案。"),
    "2512.21711": ("这篇非常关键，因为它直接挑战 latent tokens 是否 faithful，且给出 causal/adversarial 证据。", "高，是你后续 stress benchmark 的直接理论支点。"),
    "2601.08058": ("Reasoning Beyond CoT 用 latent feature steering 说明 reasoning 不必依赖显式 verbal CoT。", "高，它支持“latent computational mode exists”这个大前提。"),
    "2602.00449": ("这篇抓住了 sequential reasoning 这个硬约束，问 latent-CoT 是否真的 step-by-step。", "高，非常贴近你要做的机制/faithfulness问题。"),
    "2602.01148": ("Fundamental Limits 是少数真正试图给 latent CoT 设定边界条件的论文。", "高，适合直接指导你避开无效命题。"),
    "2602.08783": ("Dynamics Within Latent CoT 用 step-wise do-intervention，把 latent process 当成可操控因果系统。", "高，这基本就是“如何把 latent reasoning 变成科学问题”的代表。"),
    "2603.03475": ("When Shallow Wins 是当前最值得警惕的负结果之一：深 latent 计算并不自动更 faithful。", "高，直接支持你做 failure-first benchmark。"),
    "2603.06222": ("SPOT 的价值在于 span-level latent reasoning，试图同时保留效率和可解释性。", "高，说明 interpretability-efficient tradeoff 已经被正面讨论。"),
}


def shorten(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def main() -> int:
    entries = {e["arxiv_id"]: e for e in json.loads(DATA.read_text())}
    lines: list[str] = []
    lines.append("# Core Pure-LLM Latent Reasoning Deep Review")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Source pool: [LLM_LATENT_LITERATURE_NOTES.md](./LLM_LATENT_LITERATURE_NOTES.md)")
    lines.append("- Original pool size: `164` papers from the `Large-Language-Model` section")
    lines.append("- Deep-review subset: `39` core papers")
    lines.append("- Filtering rule: remove multimodal/VLM/VLA spillover, recommendation/retrieval/code/chemistry applications, agent-heavy communication work, and safety-only branches; retain papers that materially shape pure LLM latent reasoning")
    lines.append("")
    lines.append("## Reading Strategy")
    lines.append("")
    lines.append("- This file is not a full-survey duplication. It is a **core-corpus deep version** for deciding research direction.")
    lines.append("- Each paper keeps four fields: `解决的问题`, `核心方法/创新`, `我的判断`, `与你的方向关系`.")
    lines.append("- The goal is to answer: what must be read carefully, what is already crowded, and where real gaps remain.")
    lines.append("")
    lines.append("## High-Level Takeaways")
    lines.append("")
    lines.append("- The crowded lines are `latent compression`, `looped / recurrent test-time scaling`, and `latent-text routing`.")
    lines.append("- The highest-value unsolved line is still `faithfulness vs shortcut compression`: whether latent tokens really perform multi-step computation or merely hide compressed answer signals.")
    lines.append("- For your resource setting, `benchmark/diagnostic first` remains stronger than `new architecture first`.")
    lines.append("")

    total = 0
    for section, intro, ids in GROUPS:
        lines.append(f"## {section}")
        lines.append("")
        lines.append(intro)
        lines.append("")
        for aid in ids:
            total += 1
            e = entries[aid]
            judgement, fit = NOTES[aid]
            lines.append(f"### {total}. {e['title']}")
            lines.append("")
            lines.append(f"- arXiv: [{aid}]({e['paper_url']})")
            lines.append(f"- Venue: {e['venue']}")
            lines.append(f"- 解决的问题: {shorten(e['problem'])}")
            lines.append(f"- 核心方法/创新: {shorten(e['innovation'], 260)}")
            lines.append(f"- 我的判断: {judgement}")
            lines.append(f"- 与你的方向关系: {fit}")
            lines.append("")

    lines.append("## Direction Decision After The Deep Cut")
    lines.append("")
    lines.append("- **最值得继续读透的机制/负结果论文**: `2504.10615`, `2512.21711`, `2602.00449`, `2602.01148`, `2602.08783`, `2603.03475`, `2603.06222`.")
    lines.append("- **最值得保留的低算力方法/基线论文**: `2412.06769`, `2502.12134`, `2502.21074`, `2502.05171`, `2502.17416`, `2510.25741`, `2602.14759`, `2602.11683`.")
    lines.append("- **最不建议单独立项的方向**: 再做一个 latent router、再做一个 latent compression trick、再做一个 latent reasoning application paper.")
    lines.append("- **最推荐的研究切口**: 基于上述机制论文构建 answer-preserving stress benchmark，再在 benchmark 上验证 small-model phase diagram 和 lightweight repair objective.")
    lines.append("")

    OUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
