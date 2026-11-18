"""Combined evidence table across retrievers, plus the two failure cases in detail."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = {s: json.load(open(os.path.join(HERE, "results", f"retrieval_{s}.json"), encoding="utf-8")) for s in ("256", "512")}
B = {s: json.load(open(os.path.join(HERE, "results", f"bm25_{s}.json"), encoding="utf-8")) for s in ("256", "512")}
qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))
D = {s: {r["id"]: r for r in R[s]["detail"]} for s in ("256", "512")}
BR = {s: {r["id"]: r for r in B[s]["rows"]} for s in ("256", "512")}

print(f"{'id':>3} {'gold':>4} | {'d256':>4} {'d512':>4} {'b256':>4} {'b512':>4}   question")
for x in qs:
    f = lambda v: ("-" if v is None else str(v))
    print(f"{x['id']:>3} {x['gold_page']:>4} | {f(D['256'][x['id']]['first_gold_rank']):>4} "
          f"{f(D['512'][x['id']]['first_gold_rank']):>4} {f(BR['256'][x['id']]['bm25']):>4} "
          f"{f(BR['512'][x['id']]['bm25']):>4}   {x['q'][:64]}")

print()
for s in ("256", "512"):
    page_but_not_chunk = [r["id"] for r in R[s]["detail"] if r["page_in_top5"] and r["first_gold_rank"] is None]
    print(f"size {s}: page_hit@5 {R[s]['page_hit_at_5']:.3f} | chunk recall {R[s]['recall_at_5']:.3f} "
          f"| right page but wrong chunk: {page_but_not_chunk}")

print("\n===== FAILURE CASE A (q9) =====")
x = next(q for q in qs if q["id"] == 9)
print("Q:", x["q"])
print("GOLD (p%d): %s" % (x["gold_page"], x["gold_quote"][:200]))
for s in ("256", "512"):
    print(f"dense {s} top5:", [(t["rank"], t["page"], t["gold"]) for t in D[s][9]["top5"]])
    print(f"   rank1: {D[s][9]['top5'][0]['text'][:160]}")
print("bm25 512 rank:", BR["512"][9]["bm25"], "| pages:", BR["512"][9]["bm25_top5_pages"])

print("\n===== FAILURE CASE B (q15) =====")
x = next(q for q in qs if q["id"] == 15)
print("Q:", x["q"])
print("GOLD (p%d): %s" % (x["gold_page"], x["gold_quote"][:200]))
for s in ("256", "512"):
    print(f"dense {s} top5:", [(t["rank"], t["page"], t["gold"]) for t in D[s][15]["top5"]])
    print(f"   rank1: {D[s][15]['top5'][0]['text'][:160]}")
print("bm25 512 rank:", BR["512"][15]["bm25"], "| pages:", BR["512"][15]["bm25_top5_pages"])
