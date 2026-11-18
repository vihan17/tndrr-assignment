import os, fitz

d = fitz.open("report.pdf")
out = "preview"
os.makedirs(out, exist_ok=True)
for i, page in enumerate(d):
    pix = page.get_pixmap(dpi=110)
    p = os.path.join(out, f"page{i+1}.png")
    pix.save(p)
    print(p, pix.width, "x", pix.height)
