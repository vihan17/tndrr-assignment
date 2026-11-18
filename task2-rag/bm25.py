"""BM25 baseline and a dense+BM25 hybrid (reciprocal rank fusion), for both chunk sizes.

This is not required by the brief, but it turns "BM25 would fix this" from an assertion into a measurement.

Output: results/bm25_<size>.json
"""
import json, os, re, sys
import numpy as np
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = re.compile(r"[a-z0-9]+")


def toks(s):
    return TOKEN.findall(s.lower())


for SIZE in ("256", "512"):
    chunks = [json.loads(l) for l in open(os.path.join(HERE, "corpus", f"chunks_{SIZE}.jsonl"), encoding="utf-8")]
    qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))
    dense = json.load(open(os.path.join(HERE, "results", f"retrieval_{SIZE}.json"), encoding="utf-8"))

    bm25 = BM25Okapi([toks(c["text"]) for c in chunks])
    idx = {c["id"]: i for i, c in enumerate(chunks)}

    def ranks_for(score_vec, k=10):
        return [chunks[int(j)]["id"] for j in np.argsort(-score_vec)[:k]]

    out = {"size": int(SIZE), "n_chunks": len(chunks), "rows": []}
    hit_d = hit_b = hit_h = 0
    rr_d = rr_b = rr_h = 0.0

    for x in qs:
        gold = [c["id"] for c in chunks if x["gold_quote"] in c["text"]]
        s = bm25.get_scores(toks(x["q"]))
        b_order = ranks_for(s)
        d_order = ranks_for(np.array([dense_rank[x["id"]].get(c["id"], 999) for c in chunks]) * -1) if False else None

        # dense order straight from the saved top5 (k=5 is all we saved)
        d_top = [t["id"] for t in sorted(dense["detail"][[r["id"] for r in dense["detail"]].index(x["id"])]["top5"], key=lambda t: t["rank"])]

        # RRF over dense top-5 + bm25 top-10
        rrf = {}
        for r, cid in enumerate(d_top, start=1):
            rrf[cid] = rrf.get(cid, 0) + 1.0 / (60 + r)
        for r, cid in enumerate(b_order, start=1):
            rrf[cid] = rrf.get(cid, 0) + 1.0 / (60 + r)
        h_top = [c for c, _ in sorted(rrf.items(), key=lambda kv: -kv[1])][:5]

        def stat(order):
            for r, cid in enumerate(order, start=1):
                if cid in gold:
                    return r
            return None

        rd, rb, rh = stat(d_top), stat(b_order), stat(h_top)
        hit_d += rd is not None
        hit_b += rb is not None
        hit_h += rh is not None
        rr_d += (1.0 / rd) if rd else 0
        rr_b += (1.0 / rb) if rb else 0
        rr_h += (1.0 / rh) if rh else 0

        out["rows"].append({"id": x["id"], "dense": rd, "bm25": rb, "hybrid": rh,
                            "bm25_top5_pages": [c["page"] for c in (chunks[idx[c]] for c in b_order[:5])]})

    n = len(qs)
    out["summary"] = {
        "dense": {"recall@5": round(hit_d / n, 4), "mrr@5": round(rr_d / n, 4)},
        "bm25": {"recall@5": round(hit_b / n, 4), "mrr@5": round(rr_b / n, 4)},
        "hybrid_rrf": {"recall@5": round(hit_h / n, 4), "mrr@5": round(rr_h / n, 4)},
    }
    json.dump(out, open(os.path.join(HERE, "results", f"bm25_{SIZE}.json"), "w", encoding="utf-8"), indent=1)

    print(f"=== chunk size {SIZE} ({len(chunks)} chunks) ===")
    for k, v in out["summary"].items():
        print(f"  {k:12s} recall@5 {v['recall@5']:.3f} | mrr@5 {v['mrr@5']:.3f}")
    print()
