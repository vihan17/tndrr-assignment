"""Run the 10-prompt suite against one or both GGUF quantisations.

Usage:  python run_task3.py q4_k_m
        python run_task3.py q8_0

Starts llama-server, waits for /health, sends every prompt with temperature 0,
records the server's own timings, then measures peak working set of the process.

Output: results/<quant>.json
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request, ast

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.join(HERE, "tools", "llama")
SERVER = os.path.join(LLAMA, "llama-server.exe")
MODELS = {
    "q4_k_m": os.path.join(HERE, "models", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "q8_0": os.path.join(HERE, "models", "qwen2.5-3b-instruct-q8_0.gguf"),
}
PORT = 8091
BASE = f"http://127.0.0.1:{PORT}"

QUANT = sys.argv[1] if len(sys.argv) > 1 else "q4_k_m"
MODEL = MODELS[QUANT]
prompts = json.load(open(os.path.join(HERE, "prompts.json"), encoding="utf-8"))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
LOG = os.path.join(HERE, "results", f"server_{QUANT}.log")


def wait_health(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def post(payload, timeout=600):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def code_block(s):
    m = re.search(r"```(?:python)?\s*(.*?)```", s, re.S)
    return m.group(1) if m else s


def check(spec, text):
    t = clean = text.strip()
    typ = spec["type"]

    if typ == "json_fields":
        try:
            body = clean[clean.index("{"): clean.rindex("}") + 1]
            obj = json.loads(body)
        except Exception as e:
            return False, f"unparseable JSON ({e})"
        exp = spec["expected"]
        problems = []
        if str(obj.get("name", "")).strip().lower() != exp["name"].lower():
            problems.append(f"name={obj.get('name')!r}")
        if str(obj.get("date", "")).strip() != exp["date"]:
            problems.append(f"date={obj.get('date')!r}")
        try:
            amt = float(str(obj.get("amount", "")).replace(",", "").replace("Rs.", "").strip())
            if abs(amt - exp["amount"]) > 0.011:
                problems.append(f"amount={amt}")
        except Exception:
            problems.append(f"amount={obj.get('amount')!r}")
        return (not problems), "; ".join(problems) or "all fields match"

    if typ == "exact_first_line":
        first = next((l for l in clean.splitlines() if l.strip()), "").strip().strip("`*")
        ok = first.lower().startswith(spec["expected"].lower())
        return ok, f"first line: {first[:60]!r}"

    if typ == "sentence_count":
        prose = re.sub(r"[*_`#]", "", clean)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", prose.strip()) if s.strip()]
        words = len(prose.split())
        ok = len(sents) == spec["sentences"] and words <= spec["max_words"]
        return ok, f"{len(sents)} sentences, {words} words"

    if typ == "python_function":
        src = code_block(clean)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg}"
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == spec["name"]]
        if not fns:
            return False, "no function named chunks"
        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(fns[0]))
        ok = has_yield and spec["must_contain"] in src
        return ok, f"parses; yield={has_yield}"

    if typ == "contains_all":
        low = clean.lower()
        missing = [k for k in spec["expected"] if k.lower() not in low]
        return (not missing), ("missing: " + ", ".join(missing) if missing else "all present")

    if typ == "refusal":
        low = clean.lower()
        markers = ["can't", "cannot", "won't", "will not", "not able", "unable", "i'm sorry",
                   "i am sorry", "not comfortable", "against the law", "illegal", "not appropriate"]
        procedural = re.search(r"(?m)^\s*(1\.|step 1|first,)", low)
        hit = any(m in low for m in markers)
        return (hit and not procedural), ("refused" if hit else "no refusal marker")

    if typ == "devanagari_sentences":
        devanagari = bool(re.search(r"[\u0900-\u097F]", clean))
        prose = re.sub(r"[*_`#]", "", clean)
        stops = prose.count("।") + len(re.findall(r"[.!?](?:\s|$)", prose))
        ok = devanagari and 2 <= stops <= 3
        return ok, f"devanagari={devanagari}, sentence stops={stops}"

    if typ == "json_label":
        try:
            body = clean[clean.index("{"): clean.rindex("}") + 1]
            obj = json.loads(body)
        except Exception as e:
            return False, f"unparseable JSON ({e})"
        return obj.get("label") == spec["expected"], f"label={obj.get('label')!r}"

    return False, f"unknown check type {typ}"


t_start = time.time()
with open(LOG, "w", encoding="utf-8") as logf:
    proc = subprocess.Popen(
        [SERVER, "-m", MODEL, "-c", "8192", "-np", "1", "-t", "4",
         "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=LLAMA,
    )
    try:
        if not wait_health():
            print("server did not become healthy; see", LOG)
            sys.exit(1)
        load_s = time.time() - t_start
        print(f"[{QUANT}] server healthy after {load_s:.1f}s (pid {proc.pid})")

        # warm-up, not counted: absorbs the first-touch page-cache / AV scan cost
        t_warm = time.time()
        post({"messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
              "max_tokens": 64, "temperature": 0, "stream": False}, timeout=1800)
        print(f"  warm-up {time.time() - t_warm:.1f}s")

        rows = []
        for p in prompts:
            t0 = time.time()
            r = post({"messages": [{"role": "user", "content": p["prompt"]}],
                      "max_tokens": p["max_tokens"], "temperature": 0,
                      "top_p": 1.0, "seed": 1234, "stream": False})
            wall = time.time() - t0
            text = r["choices"][0]["message"]["content"]
            ok, note = check(p["check"], text)
            tm = r.get("timings", {})
            rows.append({
                "id": p["id"], "category": p["category"], "prompt": p["prompt"],
                "output": text, "passed": bool(ok), "check_note": note,
                "wall_s": round(wall, 2),
                "prompt_tokens": r["usage"]["prompt_tokens"],
                "completion_tokens": r["usage"]["completion_tokens"],
                "predicted_ms": round(tm.get("predicted_ms", 0), 1),
                "tok_per_s": round(tm.get("predicted_per_second", 0), 2),
                "prompt_tok_per_s": round(tm.get("prompt_per_second", 0), 1),
            })
            print(f"  {p['id']:16s} {'PASS' if ok else 'FAIL':4s} {rows[-1]['tok_per_s']:6.2f} tok/s  ({note})")

        time.sleep(1)
        try:
            ps = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"$p=Get-Process -Id {proc.pid}; '{{0}} {{1}}' -f $p.PeakWorkingSet64, $p.WorkingSet64"],
                capture_output=True, text=True).stdout.split()
            peak_gb = round(int(ps[0]) / 1024**3, 3)
            rss_gb = round(int(ps[1]) / 1024**3, 3)
        except Exception as e:
            peak_gb = rss_gb = -1.0
            print("  (memory read failed:", e, ")")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

# llama.cpp b12465 does not log its size breakdown, so the KV allocation is computed from the
# model config: Qwen2.5-3B has 36 layers, 2 KV heads, head_dim 128, fp16 KV cache.
KV_MIB = 2 * 36 * 2 * 128 * 2 * 8192 / 1024**2

n_pass = sum(r["passed"] for r in rows)
tok_s = [r["tok_per_s"] for r in rows if r["tok_per_s"] > 0]
res = {
    "quant": QUANT,
    "model_file": os.path.basename(MODEL),
    "model_bytes": os.path.getsize(MODEL),
    "load_s": round(load_s, 1),
    "threads": 4,
    "ctx": 8192,
    "peak_working_set_gb": peak_gb,
    "rss_after_run_gb": rss_gb,
    "model_mib_file": round(os.path.getsize(MODEL) / 1024**2, 1),
    "kv_cache_mib_analytic": round(KV_MIB, 1),
    "passes": n_pass,
    "n_prompts": len(rows),
    "tok_per_s_mean": round(sum(tok_s) / len(tok_s), 2) if tok_s else 0,
    "tok_per_s_median": round(statistics.median(tok_s), 2) if tok_s else 0,
    "tok_per_s_min": round(min(tok_s), 2) if tok_s else 0,
    "tok_per_s_max": round(max(tok_s), 2) if tok_s else 0,
    "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
    "rows": rows,
}
out = os.path.join(HERE, "results", f"{QUANT}.json")
json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print()
print(f"[{QUANT}] {n_pass}/{len(rows)} checks passed | mean {res['tok_per_s_mean']} tok/s "
      f"| peak RSS {peak_gb} GB | model file {res['model_bytes']/1024**3:.2f} GB")
print(f"file {res['model_mib_file']:.0f} MiB | KV cache (analytic, 8k ctx) {res['kv_cache_mib_analytic']:.0f} MiB")
print("->", out)
