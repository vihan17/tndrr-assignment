"""Probe the extracted corpus for exact sentences, so question gold-quotes are verbatim."""
import json, re

PAGES = r"C:\Users\Chetan\tndrr-assignment\task2-rag\corpus\pages.jsonl"
pages = [json.loads(l) for l in open(PAGES, encoding="utf-8")]

TERMS = [
    "Risk tolerance",
    "four specific functions",
    "TEVV",
    "cross-cutting function",
    "AI RMF Playbook",
    "three major categories of bias",
    "Profiles",
    "AI actors",
    "effectiveness of the AI RMF",
    "How AI Risks Differ from Traditional Software Risks",
    "human-AI interaction",
    "designed, developed, deployed",
    "risk measurement",
    "documentation",
]


def ctx(text, i, before=160, after=340):
    return text[max(0, i - before): i + after].strip()


for term in TERMS:
    hits = 0
    print("=" * 100)
    print("TERM:", term)
    for p in pages:
        for m in re.finditer(re.escape(term), p["text"], flags=re.I):
            print(f"  p{p['page']:02d}: ...{ctx(p['text'], m.start())}...")
            hits += 1
            break
        if hits >= 2:
            break
    if not hits:
        print("  (no match)")
