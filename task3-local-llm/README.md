# Task 3 — local model, two quantisation settings

Qwen2.5-3B-Instruct in GGUF, served by **llama.cpp b12465** (CPU build) on an i3-1115G4
(2 cores / 4 threads, 7.7 GB RAM). Two settings: **Q4_K_M** and **Q3_K_M**.
Q8_0 and Q6_K were attempted and rejected — see below.
Context 8,192, temperature 0, seed 1234, 4 threads.

Ten prompts, each with a pass/fail check rather than a subjective score: JSON extraction,
arithmetic, a two-sentence summary, code that must parse, factual recall, a refusal, Hindi,
classification, a syllogism, and long-context recall.

## Results

| Metric | Q4_K_M | Q3_K_M |
|---|---:|---:|
| File size | 2,007 MiB (1.96 GiB) | 1,644 MiB (1.61 GiB) |
| Nominal bits / weight | 4–6 mixed | 3–5 mixed |
| KV cache at 8k ctx (fp16) | 288 MiB | 288 MiB |
| Peak working set (measured) | **1.86 GB** | **1.54 GB** |
| Load time | 38.8 s | 41.2 s |
| Decode speed, median | **9.12 tok/s** | **11.46 tok/s** |
| Decode speed, range | 0.74 – 10.29 tok/s | 2.88 – 12.14 tok/s |
| Checks passed | **9/10** | **8/10** |

The tok/s spread is real: two physical cores that are also running Windows and an IDE. The medians
are the headline; the means (7.41 and 9.94) are dragged down by one long answer each (Hindi on Q4,
the refusal on Q3). The trade is clean: Q3_K_M is 1.26× faster and 1.20× smaller in peak RSS. On
quality it loses one check relative to Q4 (`summary_len`: 3 sentences / 50 words instead of 2 /
≤40); the Hindi failure is common to both, so it is not a quantisation effect.

## Why not Q8_0 or Q6_K

Q8_0 (3.37 GiB) loaded, but the first forward pass never completed: 17 minutes of warm-up at ~7%
CPU, free physical memory down to 0.06 GB. Q6_K (2.60 GiB) did warm up, but at **0.06 tok/s** —
875.7 s for 51 tokens, 17.5 s per token. Both were paging, not computing, so I killed them. The
ceiling on this 7.7 GB laptop sits between Q4_K_M (1.86 GB peak working set, 9.1 tok/s) and Q6_K,
so the second setting has to be *below* Q4. Q3_K_M (~1.6 GiB) is the next rung down that fits with
headroom. Logs for both rejected attempts: `results/server_q8_0.log`,
`results/server_q6_k.log`. On a constrained host, the quantisation you can run is set by the
memory ceiling, not by the quality ladder.

## Reproduce

Model files are **not** committed (2.0 GiB + 1.6 GiB). Fetch them into `models/`:

```powershell
curl.exe -L -o models/qwen2.5-3b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
curl.exe -L -o models/qwen2.5-3b-instruct-q3_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q3_k_m.gguf"
```

Binaries: llama.cpp release `b12465`, `llama-b12465-bin-win-cpu-x64.zip`, extracted into
`tools/llama/`.

```powershell
python run_task3.py q4_k_m     # starts llama-server, warms up, runs the 10 prompts
python run_task3.py q3_k_m
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
| `summarize.py` | Q4 vs Q3 table from the result JSONs |
| `results/q4_k_m.json`, `results/q3_k_m.json` | full outputs, per-prompt tok/s, memory |
| `results/server_*.log` | raw llama-server logs (including the rejected Q8/Q6 attempts) |
| `REPORT_task3.md` | KV cache, quantisation memory maths, 20-user / 24 GB serving plan |

## Two numbers worth checking by hand

**KV cache.** `2 (K,V) × 36 layers × 2 KV heads × 128 head_dim × 2 bytes = 36 KiB/token`;
at 8,192 tokens that is **288 MiB**. Identical for both quants, because quantisation changes the
weights, not the cache. Grouped-query attention (16 Q heads share 2 KV heads) already cuts it 8×.

**7B on 24 GB, 20 users.** `2 × 28 × 4 × 128 × 2 = 56 KiB/token`; 8k context = 448 MiB/user;
20 users = 8.75 GiB; plus ~4.1 GiB weights (Q4) and ~2 GiB overhead ≈ **15 GiB of 24**. With an
8-bit KV cache the KV term halves and the total is ~10.5 GiB.
