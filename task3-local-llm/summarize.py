"""Compare the quantisation runs and write results/comparison.json."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["q4_k_m", "q3_k_m"]
QUANTS = [q for q in ORDER if os.path.exists(os.path.join(HERE, "results", f"{q}.json"))]

if len(QUANTS) < 2:
    print("need at least two runs; have:", QUANTS)
    raise SystemExit(0)

R = {q: json.load(open(os.path.join(HERE, "results", f"{q}.json"), encoding="utf-8")) for q in QUANTS}
base, other = QUANTS[0], QUANTS[1]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


hdr = f"{'metric':28s}" + "".join(f"{q:>14s}" for q in QUANTS)
print(hdr)
print("-" * len(hdr))
rows = [
    ("model file (GB)", lambda r: f"{r['model_bytes']/1024**3:.2f}"),
    ("model file (MiB)", lambda r: f"{r['model_mib_file']:.0f}"),
    ("effective bits/weight", lambda r: f"{r['model_bytes'] * 8 / 3.09e9:.2f}"),
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
print(f"{'prompt':16s} {base:>8s} {other:>8s} {base + ' t/s':>11s} {other + ' t/s':>11s}  note")
print("-" * 86)
per = {}
same = 0
for a, b in zip(R[base]["rows"], R[other]["rows"]):
    ident = norm(a["output"]) == norm(b["output"])
    same += ident
    per[a["id"]] = {f"{base}_pass": a["passed"], f"{other}_pass": b["passed"],
                    f"{base}_tok_s": a["tok_per_s"], f"{other}_tok_s": b["tok_per_s"],
                    "identical_output": ident,
                    f"{base}_note": a["check_note"], f"{other}_note": b["check_note"]}
    print(f"{a['id']:16s} {str(a['passed']):>8s} {str(b['passed']):>8s} "
          f"{a['tok_per_s']:11.2f} {b['tok_per_s']:11.2f}  {'same' if ident else 'differs'}")

print()
print(f"identical outputs: {same}/{len(per)}")
print(f"pass delta ({other} - {base}): {R[other]['passes'] - R[base]['passes']}")
print(f"speed ratio {base}/{other}: {R[base]['tok_per_s_mean'] / R[other]['tok_per_s_mean']:.2f}x")
print(f"memory ratio {base}/{other} peak RSS: "
      f"{R[base]['peak_working_set_gb'] / R[other]['peak_working_set_gb']:.2f}x")

json.dump({"summary": {q: {k: v for k, v in R[q].items() if k != "rows"} for q in QUANTS},
           "per_prompt": per, "identical_outputs": same},
          open(os.path.join(HERE, "results", "comparison.json"), "w", encoding="utf-8"), indent=1)
print("\n-> results/comparison.json")
