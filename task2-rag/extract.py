"""Extract the NIST AI RMF PDF into per-page text and print an outline.

Output: corpus/pages.jsonl  {"page": 1, "text": "..."}
Text is whitespace-normalised (single spaces) so gold-quote matching is exact and robust.
"""
import json, os, re
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "corpus", "nist-ai-rmf-1.0.pdf")
OUT = os.path.join(HERE, "corpus", "pages.jsonl")


def norm(s):
    s = s.replace("\u00ad", "")          # soft hyphen
    s = re.sub(r"-\s*\n\s*", "", s)      # hyphenated line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip()


doc = fitz.open(PDF)
rows = []
with open(OUT, "w", encoding="utf-8") as f:
    for i, page in enumerate(doc):
        text = norm(page.get_text())
        rows.append({"page": i + 1, "text": text})
        f.write(json.dumps({"page": i + 1, "text": text}) + "\n")

print(f"pages written: {len(rows)}  ->  {OUT}")
print(f"total chars: {sum(len(r['text']) for r in rows)}")
print()
for r in rows:
    head = r["text"][:130]
    print(f"p{r['page']:02d} [{len(r['text']):5d}] {head}")
