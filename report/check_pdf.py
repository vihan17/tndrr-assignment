import sys, fitz

p = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
d = fitz.open(p)
print("pages:", d.page_count)
for i, page in enumerate(d):
    t = page.get_text().strip().replace("\n", " ")
    print(f"  p{i+1}: {len(t):5d} chars | {t[:95]}")
