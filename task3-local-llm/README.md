# Task 3 — local model, two quantisation settings

Qwen2.5-3B-Instruct in GGUF, served by **llama.cpp b12465** (CPU build) on an i3-1115G4
(2 cores / 4 threads, 7.7 GB RAM). Two settings: **Q4_K_M** and **Q8_0**.
Context 8,192, temperature 0, seed 1234, 4 threads.

Ten prompts, each with a pass/fail check rather than a subjective score: JSON extraction,
arithmetic, a two-sentence summary, code that must parse, factual recall, a refusal, Hindi,
classification, a syllogism, and long-context recall.

## Results

| Metric | Q4_K_M | Q8_0 |
|---|---:|---:|
| File size | 2,007 MiB (1.96 GB) | 3,449 MiB (3.37 GB) |
| Effective bits / weight | ~5.2 | ~8.9 |
| KV cache at 8k ctx (fp16) | 288 MiB | 288 MiB |
| Peak working set (measured) | **1.86 GB** | _(pending)_ |
| Load time | 38.8 s | _(pending)_ |
| Decode speed, median | **9.12 tok/s** | _(pending)_ |
| Decode speed, range | 0.74 – 10.29 tok/s | _(pending)_ |
| Checks passed | **9/10** | _(pending)_ |

The tok/s spread is real: two physical cores that are also running Windows and an IDE. The median
is the headline; the mean (7.41) is dragged down by a long Hindi answer. The one Q4 failure is the
Hindi prompt — correct language, one sentence where two were demanded. Instruction-following, not
knowledge.

## Reproduce

Model files are **not** committed (2.0 GB + 3.4 GB). Fetch them into `models/`:

```powershell
curl.exe -L -o models/qwen2.5-3b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
curl.exe -L -o models/qwen2.5-3b-instruct-q8_0.gguf   "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf"
```

Binaries: llama.cpp release `b12465`, `llama-b12465-bin-win-cpu-x64.zip`, extracted into
`tools/llama/`.

```powershell
python run_task3.py q4_k_m     # starts llama-server, warms up, runs the 10 prompts
python run_task3.py q8_0
python summarize.py            # side-by-side table from results/*.json
```

`run_task3.py` reads decode speed from the server's own `timings` block (predicted tokens divided
by predicted milliseconds), samples the process working set for peak RSS, and excludes an explicit
warm-up request — the first forward pass after loading took **339 s** on this machine (cold page
cache), which would otherwise poison the first measured prompt.

## Files

| File | What it is |
|---|---|
| `run_task3.py` | server lifecycle, health check, warm-up, timed prompts, peak RSS |
| `prompts.json` | the ten prompts and their programmatic checks |
| `summarize.py` | Q4 vs Q8 table from the result JSONs |
| `results/q4_k_m.json`, `results/q8_0.json` | full outputs, per-prompt tok/s, memory |
| `results/server_*.log` | raw llama-server logs |
| `REPORT_task3.md` | KV cache, quantisation memory maths, 20-user / 24 GB serving plan |

## Two numbers worth checking by hand

**KV cache.** `2 (K,V) × 36 layers × 2 KV heads × 128 head_dim × 2 bytes = 36 KiB/token`;
at 8,192 tokens that is **288 MiB**. Identical for both quants, because quantisation changes the
weights, not the cache. Grouped-query attention (16 Q heads share 2 KV heads) already cuts it 8×.

**7B on 24 GB, 20 users.** `2 × 28 × 4 × 128 × 2 = 56 KiB/token`; 8k context = 448 MiB/user;
20 users = 8.75 GiB; plus ~4.1 GiB weights (Q4) and ~2 GiB overhead ≈ **15 GiB of 24**. With an
8-bit KV cache the KV term halves and the total is ~10.5 GiB.
