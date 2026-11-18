"""Embed chunks, retrieve top-5 for each question, score recall@5 and MRR@5.

Usage:  python search.py 256
        python search.py 512

Output: results/emb_<size>.npy, results/retrieval_<size>.json
"""
import json, os, sys, time
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = sys.argv[1] if len(sys.argv) > 1 else "256"
TOP_K = 5
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunks = [json.loads(l) for l in open(os.path.join(HERE, "corpus", f"chunks_{SIZE}.jsonl"), encoding="utf-8")]
questions = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
emb_path = os.path.join(HERE, "results", f"emb_{SIZE}.npy")

t0 = time.time()
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
if os.path.exists(emb_path):
    E = np.load(emb_path)
    print(f"loaded cached embeddings {E.shape}")
else:
    E = model.encode([c["text"] for c in chunks], batch_size=32,
                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.save(emb_path, E)
    print(f"embedded {E.shape} in {time.time()-t0:.1f}s")

Q = model.encode([QUERY_PREFIX + x["q"] for x in questions],
                 normalize_embeddings=True, batch_size=16).astype(np.float32)

S = Q @ E.T                      # cosine, both sides normalised

# gold chunks: those whose text contains the verbatim gold quote
for x in questions:
    x["gold_ids"] = [c["id"] for c in chunks if x["gold_quote"] in c["text"]]
    if not x["gold_ids"]:
        x["gold_ids"] = []       # flagged below

miss = [x["id"] for x in questions if not x["gold_ids"]]
if miss:
    print("WARNING: gold quote not contained in any chunk for questions:", miss)

rows, hits, rr = [], 0, 0.0
for i, x in enumerate(questions):
    order = np.argsort(-S[i])[:TOP_K]
    top = []
    for rank, j in enumerate(order):
        c = chunks[int(j)]
        is_gold = c["id"] in x["gold_ids"]
        top.append({"rank": rank + 1, "id": c["id"], "page": c["page"],
                    "score": round(float(S[i][int(j)]), 4), "gold": is_gold,
                    "text": c["text"][:220]})
    first = next((t["rank"] for t in top if t["gold"]), None)
    if first:
        hits += 1
        rr += 1.0 / first
    rows.append({"id": x["id"], "q": x["q"], "gold_page": x["gold_page"],
                 "gold_ids": x["gold_ids"], "first_gold_rank": first,
                 "page_in_top5": any(t["page"] == x["gold_page"] for t in top),
                 "top5": top})

res = {
    "size": int(SIZE),
    "n_chunks": len(chunks),
    "n_questions": len(questions),
    "recall_at_5": round(hits / len(questions), 4),
    "mrr_at_5": round(rr / len(questions), 4),
    "page_hit_at_5": round(sum(r["page_in_top5"] for r in rows) / len(rows), 4),
    "detail": rows,
}
out = os.path.join(HERE, "results", f"retrieval_{SIZE}.json")
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)

print()
print(f"chunk size      : {SIZE} tokens ({len(chunks)} chunks)")
print(f"recall@5        : {res['recall_at_5']:.4f}")
print(f"MRR@5           : {res['mrr_at_5']:.4f}")
print(f"gold page in @5 : {res['page_hit_at_5']:.4f}")
print(f"-> {out}")
