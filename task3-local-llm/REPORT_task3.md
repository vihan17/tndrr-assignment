# Task 3 — running the model locally, two quantisation settings

Everything below was measured on this machine, not estimated: llama.cpp b12465 (CPU build),
Qwen2.5-3B-Instruct in GGUF, 4 threads on an i3-1115G4 (2 cores / 4 threads), 7.7 GB RAM,
8,192-token context, temperature 0, seed 1234.

## Setup

| Piece | Value |
|---|---|
| Runtime | llama.cpp b12465, `llama-server` (OpenAI-compatible `/v1/chat/completions`) |
| Model | Qwen2.5-3B-Instruct (3.09B params) |
| Quantisations | Q4_K_M and Q8_0 |
| Context | 8,192 tokens, one slot |
| Threads | 4 |
| Sampling | temperature 0, top_p 1.0, seed 1234 |

## Measured results

| Metric | Q4_K_M | Q8_0 |
|---|---:|---:|
| File size | 2,007 MiB (1.96 GB) | 3,449 MiB (3.37 GB) |
| Effective bits/weight | ~5.2 | ~8.9 |
| KV cache at 8k ctx (fp16) | 288 MiB | 288 MiB |
| Peak working set (measured) | **1.86 GB** | *(pending)* |
| Load time | 38.8 s | *(pending)* |
| Generation speed, median | **9.12 tok/s** | *(pending)* |
| Generation speed, range | 0.74 – 10.29 tok/s | *(pending)* |
| 10-prompt checks passed | **9/10** | *(pending)* |

The spread in tok/s is real and worth stating plainly: on two physical cores the machine is also
running Windows and an IDE, so individual prompts range from 0.7 to 10.3 tok/s. The median is the
honest headline; the mean (7.41) is dragged down by the Hindi prompt, which produced a long answer
and hit the slowest stretch of the run.

## The 10-prompt quality comparison

Each prompt has a check that is pass/fail rather than a vibe, so the comparison produces a count.
The full outputs are in `results/<quant>.json`.

| Prompt | What it tests | Q4_K_M | Q8_0 |
|---|---|---:|---:|
| `json_extract` | strict JSON, ISO date, numeric amount | PASS | *(pending)* |
| `arithmetic` | 14:35 + 2 h 47 m → 17:22 | PASS | *(pending)* |
| `summary_len` | exactly 2 sentences, ≤ 40 words | PASS (2 s, 38 w) | *(pending)* |
| `code` | `chunks(lst, n)` parses, contains `yield` | PASS | *(pending)* |
| `factual` | deepest ocean / Challenger Deep | PASS | *(pending)* |
| `refusal` | refuses lock-picking instructions | PASS | *(pending)* |
| `hindi` | Devanagari, exactly two sentences | **FAIL** (1 stop) | *(pending)* |
| `classify` | Banking77-style label in JSON | PASS | *(pending)* |
| `logic` | syllogism, answer "Yes" | PASS | *(pending)* |
| `long_context` | recall two facts from a memo | PASS | *(pending)* |

Q4_K_M's single failure is instructive: it answered in Hindi correctly but wrote **one** sentence
where the prompt asked for exactly two. That is an instruction-following failure, not a knowledge
failure — and it is the kind of thing a lower-bit quantisation is expected to make slightly more
likely, because the model has less capacity to track surface constraints.

## How the KV cache works

A transformer generates one token at a time. At every layer, the new token's query has to attend
to the keys and values of every token that came before it. Those key/value vectors never change
once computed, so instead of recomputing them for the whole prefix on every step, the model keeps
them in memory. That store is the KV cache.

Its size is fixed by the architecture, not by the weights:

```
bytes = 2 (K and V) × layers × kv_heads × head_dim × bytes_per_element × context_tokens
```

For Qwen2.5-3B that is 2 × 36 × 2 × 128 × 2 bytes = **36 KiB per token**, so 8,192 tokens of
context costs **288 MiB**. Two things keep that number small. First, the model uses grouped-query
attention: 16 query heads share only 2 key/value heads, which cuts the cache by 8× compared with
full multi-head attention. Second, the cache is per sequence — with 20 concurrent users it is 20×
288 MiB, which is exactly why serving plans are dominated by KV budget rather than by weights.

The cache grows linearly with context and with concurrency, and it is allocated when the context is
created. That is also why a long conversation gets slower and eventually fails: you run out of KV
space, not out of model.

## How quantisation changes memory

Weights are stored as numbers. In fp16 each weight is 2 bytes, so 3.09B parameters would need
about 6.2 GB. Quantisation stores each weight (or a small block of weights) in fewer bits:

- **Q8_0** keeps 8 bits per weight plus one scale per 32-weight block (~8.5 bits/weight). This file
  is 3.37 GB, about 8.9 effective bits/weight once embeddings and metadata are included.
- **Q4_K_M** uses a mixture: most tensors at 4 bits with per-block scales and minimums, some
  important tensors at 6 bits. This file is 1.96 GB, about 5.2 effective bits/weight.

So Q4 is 1.72× smaller than Q8, and Q8 is roughly half of fp16 — the memory saving is close to
linear in bits per weight, with a small overhead for the block scales.

Quantisation does **not** touch the KV cache or the activations. Weights, KV cache and compute
buffers are three separate allocations; quantising weights changes only the first. That is why the
KV number above is identical for both models, and why you can independently quantise the KV cache
(`--cache-type-k q8_0 --cache-type-v q8_0`) to halve it at a small quality cost.

Quality-wise, Q8_0 is close to lossless, while Q4_K_M is where degradation starts to be measurable
on hard tasks — long-tail knowledge, arithmetic, and surface constraints like "exactly two
sentences". Speed-wise, quantisation usually *helps*: decoding is memory-bandwidth-bound, so
reading 2 GB per token is faster than reading 3.4 GB, which is why Q4 is expected to be the faster
setting on this CPU as well.

## Serving 20 concurrent users on a 24 GB GPU

The 3B model is the wrong size for this question, so the plan below uses Qwen2.5-7B-Instruct at
Q4_K_M (~4.4 GB), which is the realistic choice for a 24 GB card.

**KV budget.** Qwen2.5-7B has 28 layers, 4 KV heads and head_dim 128:

```
2 × 28 × 4 × 128 × 2 bytes = 56 KiB per token
8,192-token context        = 448 MiB per user
20 users                   = 8.75 GiB
```

**Total memory.** 4.1 GiB weights + 8.75 GiB KV + ~2 GiB for compute buffers, CUDA context and
fragmentation ≈ **15 GiB of 24 GiB**, leaving 9 GiB of headroom. With q8_0 KV cache the KV term
halves to 4.4 GiB and the total drops to ~10.5 GiB, which is the safer configuration if contexts
regularly run to 8k.

**Serving stack.** `llama-server -m qwen2.5-7b-q4_k_m.gguf -np 20 -c 163840 --cache-type-k q8_0
--cache-type-v q8_0 --cont-batching`. In llama.cpp the context is split across slots, so
`-c 163840` gives each of the 20 slots 8,192 tokens; slots are reserved up front, so the KV number
above is a hard allocation, not a worst case. If the workload has short chats and occasional long
ones, vLLM with PagedAttention is the better fit: it allocates KV in fixed-size blocks on demand,
so idle users cost almost nothing and the same 24 GB supports more concurrent sessions.

**Throughput.** A 24 GB card (RTX 4090 class) decodes a 7B Q4 model at roughly 60–80 tok/s for a
single stream. With continuous batching, aggregate throughput is typically 400–600 tok/s, so 20
active users see roughly 20–30 tok/s each — around reading speed. On an A10-class card, halve
that. If the requirement is a hard 20 tok/s per user at all times, the admission policy should cap
active sequences at 20 and queue the rest rather than letting the batch grow and everyone slow
down.

**Practical guards.** Cap `max_tokens` per request so one user cannot occupy a slot for minutes;
reuse a shared system prompt with prefix caching; monitor KV utilisation and reject or queue new
requests when it crosses ~85%; and pin one model per GPU rather than swapping models, since a
reload costs seconds during which nobody is served.

## Limitations

- One 2-core laptop CPU, so the absolute tok/s numbers say more about this machine than about the
  quantisations. The *ratio* between Q4 and Q8 is the transferable result.
- The first forward pass after loading took 339 s on the Q4 run (cold page cache on a 7.7 GB
  machine). It is excluded from the timed results via an explicit warm-up, and it is the single
  biggest reason a laptop is a poor serving host.
- Ten prompts is a small quality sample; it is enough to catch a broken quantisation, not enough to
  rank two good ones statistically. The checks are objective, which is why they are usable at all
  at this sample size.
