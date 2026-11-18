"""Compare the two quantisation runs and write results/comparison.json."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
QUANTS = ["q4_k_m", "q8_0"]
R = {}
for q in QUANTS:
    p = os.path.join(HERE, "results", f"{q}.json")
    if os.path.exists(p):
        R[q] = json.load(open(p, encoding="utf-8"))

if len(R) < 2:
    print("need both runs; have:", list(R))
    raise SystemExit(0)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


hdr = f"{'metric':28s}" + "".join(f"{q:>14s}" for q in QUANTS)
print(hdr)
print("-" * len(hdr))
rows = [
    ("model file (GB)", lambda r: f"{r['model_bytes']/1024**3:.2f}"),
    ("model file (MiB)", lambda r: f"{r['model_mib_file']:.0f}"),
    ("KV cache @8k (MiB)", lambda r: f"{r['kv_cache_mib_analytic']:.0f}"),
    ("peak RSS (GB)", lambda r: f"{r['peak_working_set_gb']:.2f}"),
    ("load (s)", lambda r: f"{r['load_s']:.1f}"),
    ("mean tok/s", lambda r: f"{r['tok_per_s_mean']:.2f}"),
    ("median tok/s", lambda r: f"{r['tok_per_s_median']:.2f}"),
    ("tok/s range", lambda r: f"{r['tok_per_s_min']:.2f}-{r['tok_per_s_max']:.2f}"),
    ("completion tokens", lambda r: f"{r['total_completion_tokens']}"),
    ("checks passed", lambda r: f"{r['passes']}/{r['n_prompts']}"),
]
for label, fn in rows:
    print(f"{label:28s}" + "".join(f"{fn(R[q]):>14s}" for q in QUANTS))

print()
print(f"{'prompt':16s} {'q4':>6s} {'q8':>6s} {'q4 t/s':>8s} {'q8 t/s':>8s}  note")
print("-" * 78)
per = {}
same = 0
for i, p in enumerate(R["q4_k_m"]["rows"]):
    a = R["q4_k_m"]["rows"][i]
    b = R["q8_0"]["rows"][i]
    ident = norm(a["output"]) == norm(b["output"])
    same += ident
    per[a["id"]] = {"q4_pass": a["passed"], "q8_pass": b["passed"],
                    "q4_tok_s": a["tok_per_s"], "q8_tok_s": b["tok_per_s"],
                    "identical_output": ident,
                    "q4_note": a["check_note"], "q8_note": b["check_note"]}
    print(f"{a['id']:16s} {str(a['passed']):>6s} {str(b['passed']):>6s} "
          f"{a['tok_per_s']:8.2f} {b['tok_per_s']:8.2f}  {'same' if ident else 'differs'}")

print()
print(f"identical outputs: {same}/{len(per)}")
print(f"pass delta (q8 - q4): {R['q8_0']['passes'] - R['q4_k_m']['passes']}")
print(f"speed ratio q4/q8: {R['q4_k_m']['tok_per_s_mean'] / R['q8_0']['tok_per_s_mean']:.2f}x")
print(f"memory ratio q4/q8 peak RSS: {R['q4_k_m']['peak_working_set_gb'] / R['q8_0']['peak_working_set_gb']:.2f}x")

json.dump({"summary": {q: {k: v for k, v in R[q].items() if k != "rows"} for q in QUANTS},
           "per_prompt": per, "identical_outputs": same}, 
          open(os.path.join(HERE, "results", "comparison.json"), "w", encoding="utf-8"), indent=1)
print("\n-> results/comparison.json")
