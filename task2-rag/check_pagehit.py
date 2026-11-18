import json, os

base = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(base, "results", "retrieval_256.json"), encoding="utf-8"))["detail"]
b = json.load(open(os.path.join(base, "results", "retrieval_512.json"), encoding="utf-8"))["detail"]

A = {r["id"]: r for r in a}
B = {r["id"]: r for r in b}
for qid in sorted(A):
    pa, pb = A[qid]["page_in_top5"], B[qid]["page_in_top5"]
    if pa != pb:
        print(f"q{qid}: 256 page_hit={pa} (gold p{A[qid]['gold_page']}, rank {A[qid]['first_gold_rank']}) "
              f"| 512 page_hit={pb} (gold p{B[qid]['gold_page']}, rank {B[qid]['first_gold_rank']})")
        print("   256 pages:", [h["page"] for h in A[qid]["top5"]])
        print("   512 pages:", [h["page"] for h in B[qid]["top5"]])
