"""Second probe: pull complete sentences for the subcategory/table facts I want to test."""
import json, re

PAGES = r"C:\Users\Chetan\tndrr-assignment\task2-rag\corpus\pages.jsonl"
pages = [json.loads(l) for l in open(PAGES, encoding="utf-8")]

TERMS = [
    ("GOVERN 1.5", 300),
    ("GOVERN 1.1", 300),
    ("MAP 5.2", 320),
    ("MEASURE 2.6", 320),
    ("MANAGE 4", 320),
    ("cross-sectoral profiles", 500),
    ("valid and reliable, safe, secure", 420),
    ("AI dimensions displayed in Figure 2", 300),
    ("The AI RMF Core provides", 400),
    ("Appendix A: Descriptions", 260),
    ("AI RMF Profiles", 500),
]

for term, span in TERMS:
    print("=" * 100)
    print("TERM:", term)
    found = False
    for p in pages:
        i = p["text"].find(term)
        if i >= 0:
            print(f"  p{p['page']:02d}: {p['text'][i:i + span]}")
            found = True
            break
    if not found:
        for p in pages:
            i = p["text"].lower().find(term.lower())
            if i >= 0:
                print(f"  p{p['page']:02d} (ci): {p['text'][i:i + span]}")
                found = True
                break
    if not found:
        print("  (no match)")
