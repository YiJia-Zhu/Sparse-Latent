#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Awesome-Latent-Space.md"
DATA_DIR = ROOT / "data"
JSON_OUT = DATA_DIR / "latent_llm_papers.json"
MD_OUT = ROOT / "LLM_LATENT_LITERATURE_NOTES.md"


def split_markdown_row(line: str) -> list[str]:
    if not line.startswith("|"):
        raise ValueError(f"not a table row: {line}")
    parts = [part.strip() for part in line.strip().split("|")[1:-1]]
    return parts


def clean_text(text: str) -> str:
    text = text.replace("<br/>", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_entry(line: str) -> dict:
    date_cell, title_cell, _intro_cell, code_cell = split_markdown_row(line)
    venue_match = re.search(r"!\[([^\]]+)\]\(", title_cell)
    venue = venue_match.group(1).strip() if venue_match else "arXiv"
    link_matches = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", title_cell)
    paper_link = None
    for label, url in link_matches:
        if "arxiv.org/abs/" in url or "arxiv.org/pdf/" in url:
            paper_link = (label, url)
            break
    if paper_link is None and link_matches:
        paper_link = link_matches[-1]
    if not paper_link:
        raise ValueError(f"missing title link: {line}")
    title = paper_link[0].strip()
    paper_url = paper_link[1].strip()
    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", paper_url)
    arxiv_id = arxiv_match.group(1) if arxiv_match else ""

    code_match = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", code_cell)
    code_url = code_match.group(2).strip() if code_match else ""

    return {
        "date": date_cell,
        "title": title,
        "paper_url": paper_url,
        "arxiv_id": arxiv_id,
        "venue": venue,
        "code_url": code_url,
    }


def parse_source(path: Path) -> list[dict]:
    text = path.read_text()
    start = text.index("### Large-Language-Model")
    end = text.index("### Vision-Language-Model")
    section = text[start:end]
    entries = []
    for line in section.splitlines():
        if line.startswith("| 20"):
            entries.append(parse_entry(line))
    return entries


def fetch_arxiv_batch(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    base = "https://export.arxiv.org/api/query"
    url = f"{base}?id_list={urllib.parse.quote(','.join(ids))}&max_results={len(ids)}"
    with urllib.request.urlopen(url, timeout=40) as resp:
        payload = resp.read()
    root = ET.fromstring(payload)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    result = {}
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.findtext("atom:id", default="", namespaces=ns)
        match = re.search(r"([0-9]+\.[0-9]+)(?:v[0-9]+)?$", raw_id)
        arxiv_id = match.group(1) if match else raw_id.rsplit("/", 1)[-1]
        authors = [
            author.findtext("atom:name", default="", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]
        categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ns)]
        result[arxiv_id] = {
            "api_title": clean_text(entry.findtext("atom:title", default="", namespaces=ns)),
            "summary": clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
            "published": entry.findtext("atom:published", default="", namespaces=ns),
            "updated": entry.findtext("atom:updated", default="", namespaces=ns),
            "authors": [a for a in authors if a],
            "comment": clean_text(entry.findtext("arxiv:comment", default="", namespaces=ns)),
            "categories": [c for c in categories if c],
        }
    return result


def fetch_arxiv_metadata(ids: list[str]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    batch_size = 20
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        try:
            metadata.update(fetch_arxiv_batch(batch))
        except urllib.error.URLError as exc:
            print(f"warning: failed to fetch batch starting at {i}: {exc}", file=sys.stderr)
        time.sleep(0.4)
    return metadata


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in pieces if p.strip()]


def choose_problem(summary: str, title: str) -> str:
    sentences = split_sentences(summary)
    if not sentences:
        return f"Inferred from the title: {title}."
    for s in sentences[:3]:
        if any(marker in s.lower() for marker in ["however", "despite", "existing", "challenge", "problem", "bottleneck", "cost", "latency", "reliability"]):
            return s
    return sentences[0]


def choose_innovation(summary: str, title: str) -> str:
    sentences = split_sentences(summary)
    for s in sentences[:5]:
        lowered = s.lower()
        if any(marker in lowered for marker in ["we propose", "we present", "our method", "we introduce", "this paper", "we find that", "we develop"]):
            return s
    if sentences:
        return sentences[min(1, len(sentences) - 1)]
    return f"Inferred from the title: {title}."


def infer_theme(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(k in text for k in ["benchmark", "analysis", "understanding", "causal", "visualizing", "characterizing", "formal comparison", "do latent tokens think", "shallow wins"]):
        return "analysis_and_diagnostics"
    if any(k in text for k in ["loop", "recurrent", "ponder", "test-time", "stop", "router", "switch", "fly", "parallel", "gradient descent"]):
        return "test_time_compute_and_routing"
    if any(k in text for k in ["distill", "softcot", "chain-of-embedding", "continuous concept", "self-distillation", "supervised thinking", "latent thoughts tuning", "soft thinking"]):
        return "latent_training_and_distillation"
    if any(k in text for k in ["latent chain", "hidden chain", "continuous latent", "latent thought", "reasoning in latent", "silent thought", "continuous chain of thought", "chain of continuous thought"]):
        return "core_latent_reasoning"
    if any(k in text for k in ["jailbreak", "safeguard", "defending", "detecting", "harmful", "unsafe", "refusal", "risk detection", "latentguard"]):
        return "safety_and_alignment"
    if any(k in text for k in ["agent", "multi-agent", "communication", "debate", "memory for agents"]):
        return "agents_and_latent_communication"
    if any(k in text for k in ["retrieval", "recommend", "recommendation", "search", "astronomer", "chem", "code language"]):
        return "applications"
    if any(k in text for k in ["cache", "kv", "compression", "efficient", "accelerating", "speed", "fast", "latency"]):
        return "efficiency_and_compression"
    return "other"


def shorten(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def build_markdown(entries: list[dict]) -> str:
    theme_counts = Counter(entry["theme"] for entry in entries)
    lines = []
    lines.append("# Pure-LLM Latent Space Literature Notes")
    lines.append("")
    lines.append(f"文献数: **{len(entries)}**")
    lines.append("范围: `Awesome-Latent-Space.md` 中 `### Large-Language-Model` 的全部条目。")
    lines.append("方法: 通过 arXiv API 批量抓取元信息和摘要，再生成逐篇的一阶阅读笔记。")
    lines.append("")
    lines.append("## Theme Summary")
    lines.append("")
    for theme, count in sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{theme}`: {count}")
    lines.append("")
    lines.append("## Quick Index")
    lines.append("")
    lines.append("| # | Date | Venue | Title | Theme |")
    lines.append("|---|---|---|---|---|")
    for idx, entry in enumerate(entries, 1):
        lines.append(
            f"| {idx} | {entry['date']} | {entry['venue']} | [{entry['title']}]({entry['paper_url']}) | `{entry['theme']}` |"
        )
    lines.append("")
    lines.append("## Paper Notes")
    lines.append("")
    for idx, entry in enumerate(entries, 1):
        authors = ", ".join(entry.get("authors", [])[:5])
        if len(entry.get("authors", [])) > 5:
            authors += ", et al."
        code_line = entry["code_url"] if entry["code_url"] else "N/A"
        lines.append(f"### {idx}. {entry['title']}")
        lines.append("")
        lines.append(f"- Date: {entry['date']}")
        lines.append(f"- Venue: {entry['venue']}")
        lines.append(f"- arXiv: {entry['arxiv_id'] or 'N/A'}")
        lines.append(f"- URL: {entry['paper_url']}")
        lines.append(f"- Code: {code_line}")
        lines.append(f"- Authors: {authors or 'N/A'}")
        lines.append(f"- Theme: `{entry['theme']}`")
        if entry.get("comment"):
            lines.append(f"- Comment: {entry['comment']}")
        lines.append(f"- 解决的问题: {shorten(entry['problem'], 280)}")
        lines.append(f"- 创新/改进: {shorten(entry['innovation'], 320)}")
        if entry.get("summary"):
            lines.append(f"- 摘要要点: {shorten(entry['summary'], 360)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    entries = parse_source(SOURCE)
    ids = [entry["arxiv_id"] for entry in entries if entry["arxiv_id"]]
    metadata = fetch_arxiv_metadata(ids)
    for entry in entries:
        meta = metadata.get(entry["arxiv_id"], {})
        entry.update(meta)
        entry["problem"] = choose_problem(entry.get("summary", ""), entry["title"])
        entry["innovation"] = choose_innovation(entry.get("summary", ""), entry["title"])
        entry["theme"] = infer_theme(entry["title"], entry.get("summary", ""))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    MD_OUT.write_text(build_markdown(entries))
    print(f"wrote {len(entries)} entries to {MD_OUT}")
    print(f"wrote structured data to {JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
