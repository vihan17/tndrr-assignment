import json, os, statistics

base = os.path.dirname(os.path.abspath(__file__))

for f in ["retrieval_256", "retrieval_512"]:
    d = json.load(open(os.path.join(base, "results", f + ".json"), encoding="utf-8"))
    det = d["detail"]
    print(f, "| n detail:", len(det), "| first keys:", list(det[0].keys()))
    print("   first entry:", json.dumps(det[0], ensure_ascii=False)[:400])

    def pages(r):
        hits = r.get("top5") or r.get("hits") or r.get("results") or []
        out = []
        for h in hits:
            if isinstance(h, dict):
                out.append(h.get("page", h.get("gold_page")))
        return out

    cov = [len(set(p for p in pages(r) if p is not None)) for r in det]
    print("   mean distinct pages in top-5:", round(statistics.mean(cov), 2),
          "| min", min(cov), "| max", max(cov))

pages = [json.loads(l) for l in open(os.path.join(base, "corpus", "pages.jsonl"), encoding="utf-8")]
print("corpus pages:", len(pages), "chars:", sum(len(p.get("text", "")) for p in pages))
