import json, os

base = os.path.dirname(os.path.abspath(__file__))
for f in ["retrieval_256", "retrieval_512", "bm25_256", "bm25_512"]:
    p = os.path.join(base, "results", f + ".json")
    d = json.load(open(p, encoding="utf-8"))
    flat = {k: v for k, v in d.items() if not isinstance(v, (list, dict))}
    print(f, "|", flat)
    if "metrics" in d:
        print("   metrics:", d["metrics"])
    if "summary" in d:
        print("   summary:", d["summary"])
