"""Chunk the corpus at two sizes, inside page boundaries so every chunk keeps a valid page number.

Output: corpus/chunks_256.jsonl and corpus/chunks_512.jsonl
Each record: {id, page, tokens, start, end, text}

Chunks are sliced on the original characters via tokenizer offset mapping, so the text is
byte-for-byte the source text (needed for exact gold-quote matching).
"""
import json, os
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [256, 512]
OVERLAP = 64

tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
pages = [json.loads(l) for l in open(os.path.join(HERE, "corpus", "pages.jsonl"), encoding="utf-8")]

for size in SIZES:
    rows = []
    for p in pages:
        text = p["text"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if not ids:
            continue
        start = 0
        while start < len(ids):
            end = min(start + size, len(ids))
            cs, ce = offs[start][0], offs[end - 1][1]
            rows.append({
                "id": f"c{size}_{len(rows):04d}",
                "page": p["page"],
                "tokens": end - start,
                "start": cs,
                "end": ce,
                "text": text[cs:ce],
            })
            if end >= len(ids):
                break
            start = end - OVERLAP
    out = os.path.join(HERE, "corpus", f"chunks_{size}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lens = [r["tokens"] for r in rows]
    print(f"size {size}: {len(rows)} chunks | tokens min {min(lens)} med {sorted(lens)[len(lens)//2]} max {max(lens)} | -> {os.path.basename(out)}")
