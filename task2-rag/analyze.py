"""Per-question comparison of the two chunk sizes, plus failure detail."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = {s: json.load(open(os.path.join(HERE, "results", f"retrieval_{s}.json"), encoding="utf-8")) for s in ("256", "512")}
D = {s: {r["id"]: r for r in R[s]["detail"]} for s in ("256", "512")}
qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))

print(f"{'id':>3} {'gold_p':>6} {'256 rank':>9} {'512 rank':>9}   question")
for x in qs:
    a = D["256"][x["id"]]["first_gold_rank"]
    b = D["512"][x["id"]]["first_gold_rank"]
    flag = "  <-- both miss" if a is None and b is None else ""
    print(f"{x['id']:>3} {x['gold_page']:>6} {str(a):>9} {str(b):>9}   {x['q'][:70]}{flag}")

print()
for s in ("256", "512"):
    r = R[s]
    print(f"size {s}: recall@5 {r['recall_at_5']:.3f} | mrr@5 {r['mrr_at_5']:.3f} | page_hit@5 {r['page_hit_at_5']:.3f} | chunks {r['n_chunks']}")

print("\n===== FAILURE DETAIL =====")
for x in qs:
    a, b = D["256"][x["id"]], D["512"][x["id"]]
    if a["first_gold_rank"] is None or b["first_gold_rank"] is None:
        print("=" * 90)
        print(f"q{x['id']} (gold p{x['gold_page']}): {x['q']}")
        print(f"  256 rank {a['first_gold_rank']} | 512 rank {b['first_gold_rank']}")
        for s, row in (("256", a), ("512", b)):
            print(f"  {s} top5 pages:", [(t["rank"], t["page"], t["gold"]) for t in row["top5"]])
            print(f"  {s} rank1: {row['top5'][0]['text'][:150]}")
