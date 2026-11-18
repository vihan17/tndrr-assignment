"""Verify every gold quote appears verbatim on the page it claims to come from."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
pages = [json.loads(l) for l in open(os.path.join(HERE, "corpus", "pages.jsonl"), encoding="utf-8")]
by_page = {p["page"]: p["text"] for p in pages}
qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))

bad = 0
for x in qs:
    txt = by_page.get(x["gold_page"], "")
    ok = x["gold_quote"] in txt
    if not ok:
        bad += 1
        # find where it actually is, to fix the citation
        where = [p["page"] for p in pages if x["gold_quote"] in p["text"]]
        print(f"FAIL q{x['id']:02d} page {x['gold_page']} | found on: {where}")
    else:
        print(f"ok   q{x['id']:02d} page {x['gold_page']}")

print()
print(f"{len(qs) - bad}/{len(qs)} verified")
