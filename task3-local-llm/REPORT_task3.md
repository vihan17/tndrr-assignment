# Task 3 — running the model locally, two quantisation settings

Everything below was measured on this machine, not estimated: llama.cpp b12465 (CPU build),
Qwen2.5-3B-Instruct in GGUF, 4 threads on an i3-1115G4 (2 cores / 4 threads), 7.7 GB RAM,
8,192-token context, temperature 0, seed 1234.

## Setup

| Piece | Value |
|---|---|
| Runtime | llama.cpp b12465, `llama-server` (OpenAI-compatible `/v1/chat/completions`) |
| Model | Qwen2.5-3B-Instruct (3.09B params) |
| Quantisations | **Q4_K_M** and **Q3_K_M** (Q8_0 and Q6_K attempted; both thrash — see below) |
| Context | 8,192 tokens, one slot |
| Threads | 4 |
| Sampling | temperature 0, top_p 1.0, seed 1234 |

## Measured results

| Metric | Q4_K_M | Q3_K_M |
|---|---:|---:|
| File size | 2,007 MiB (1.96 GiB) | 1,644 MiB (1.61 GiB) |
| Nominal bits / weight | 4–6 mixed | 3–5 mixed |
| KV cache at 8k ctx (fp16) | 288 MiB | 288 MiB |
| Peak working set (measured) | **1.86 GB** | **1.54 GB** |
| Load time | 38.8 s | 41.2 s |
| Generation speed, median | **9.12 tok/s** | **11.46 tok/s** |
| Generation speed, range | 0.74 – 10.29 tok/s | 2.88 – 12.14 tok/s |
| 10-prompt checks passed | **9/10** | **8/10** |

The spread in tok/s is real and worth stating plainly: on two physical cores the machine is also
running Windows and an IDE, so individual prompts range from 0.7 to 10.3 tok/s on Q4_K_M and 2.9 to
12.1 tok/s on Q3_K_M. The median is the honest headline; the means (7.41 and 9.94) are dragged down
by one long answer each (Hindi on Q4, the refusal on Q3). The trade is clean: Q3_K_M is **1.26×
faster** and **1.20× smaller** in peak RSS, and it loses exactly one additional check.

## Why the second setting is Q3_K_M, not Q8_0 or Q6_K

My first plan was Q4_K_M vs Q8_0. The Q8_0 file is 3,449 MiB (3.37 GiB), and llama.cpp loaded it
(`model loaded` 82 s after launch), but the first forward pass never finished: 17 minutes of warm-up
at ~7% CPU, working set oscillating between 1.1 and 1.5 GB, and free physical memory down to
0.06 GB. I killed it.

The next rung down, Q6_K, is 2,664 MiB (2.60 GiB). It loaded in 38.8 s and did complete the warm-up
— but the warm-up is the evidence: **51 generated tokens in 875.7 s, or 0.06 tok/s**, with prompt
eval at 2.29 tok/s. At 17.5 s per token, one 64-token answer would take ~19 minutes. That is
paging, not compute.

So the ceiling on this 7.7 GB laptop sits **between Q4_K_M** (2,007 MiB file, 1.86 GB peak working
set, 9.1 tok/s) **and Q6_K**. The usable second setting has to be *below* Q4, not above it. I picked
**Q3_K_M** (~1.6 GiB), the next rung down with clear headroom. Both rejected attempts left server
logs (`results/server_q8_0.log`, `results/server_q6_k.log`), so the claim is checkable rather than a
story.

That is the real lesson of this task: on a constrained host, the quantisation you can run is set by
the memory ceiling, not by the quality ladder. A Q8_0 number produced by a machine that is
thrashing would be a number about the page file, not about the model. The comparison below therefore
measures the cliff from below: Q4_K_M vs Q3_K_M.

## The 10-prompt quality comparison

Each prompt has a check that is pass/fail rather than a vibe, so the comparison produces a count.
The full outputs are in `results/<quant>.json`.

| Prompt | What it tests | Q4_K_M | Q3_K_M |
|---|---|---:|---:|
| `json_extract` | strict JSON, ISO date, numeric amount | PASS | PASS |
| `arithmetic` | 14:35 + 2 h 47 m → 17:22 | PASS | PASS |
| `summary_len` | exactly 2 sentences, ≤ 40 words | PASS (2 s, 38 w) | **FAIL** (3 s, 50 w) |
| `code` | `chunks(lst, n)` parses, contains `yield` | PASS | PASS |
| `factual` | deepest ocean / Challenger Deep | PASS | PASS |
| `refusal` | refuses lock-picking instructions | PASS | PASS |
| `hindi` | Devanagari, exactly two sentences | **FAIL** (1 stop) | **FAIL** (1 stop) |
| `classify` | Banking77-style label in JSON | PASS | PASS |
| `logic` | syllogism, answer "Yes" | PASS | PASS |
| `long_context` | recall two facts from a memo | PASS | PASS |

Q4_K_M's single failure is instructive: it answered in Hindi correctly but wrote **one** sentence
where the prompt asked for exactly two. That is an instruction-following failure, not a knowledge
failure. Q3_K_M is the direct test of whether a lower-bit setting makes that kind of surface
constraint failure more likely.

Q3_K_M answers that test: it loses one check relative to Q4_K_M. On `summary_len` it wrote **3
sentences and 50 words** instead of exactly 2 and ≤40 — the same class of failure (surface
constraint, not knowledge) and the expected direction. Everything else held: JSON, arithmetic,
code, factual recall, refusal, classification, logic and long-context recall all passed. The Hindi
prompt failed in both settings, so that one is about the prompt/model, not the quantisation.

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

- **Q8_0** (attempted, does not fit) keeps 8 bits per weight plus one scale per 32-weight block.
  Its file is 3.37 GiB, roughly half of fp16. It loaded, but never completed a forward pass.
- **Q6_K** (attempted, does not fit) keeps 6 bits per weight with 8-bit block scales. Its file is
  2.60 GiB and it generated at 0.06 tok/s — it fits in the address space but not in RAM.
- **Q4_K_M** uses a mixture: most tensors at 4 bits with per-block scales and minimums, some
  important tensors at 6 bits. Its file is 1.96 GiB, and it is the first setting that runs at a
  usable speed.
- **Q3_K_M** keeps ~3 bits per weight, with a few 4- and 5-bit tensors. Its file is ~1.6 GiB, the
  second setting that fits with headroom.

So the file sizes track bits per weight almost linearly: Q4 is about 1.22× the size of Q3, Q6 is
about 1.33× the size of Q4, and Q8 is about 1.29× the size of Q6. The saving is close to linear
with a small overhead for the block scales.

Quantisation does **not** touch the KV cache or the activations. Weights, KV cache and compute
buffers are three separate allocations; quantising weights changes only the first. That is why the
KV number above is identical for both models, and why you can independently quantise the KV cache
(`--cache-type-k q8_0 --cache-type-v q8_0`) to halve it at a small quality cost.

Quality-wise, the standard expectation is that Q8_0 and Q6_K are close to lossless and Q4/Q3 are
where long-tail knowledge and surface constraints start to wobble. This machine cannot test that
ladder from above, so the comparison is the Q4→Q3 step. Speed-wise, quantisation usually *helps*:
decoding is memory-bandwidth-bound, so reading 1.6 GiB per token should be faster than reading
1.96 GiB, which is why Q3 is expected to be the faster setting on this CPU.

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
  quantisations. The *ratio* between Q4 and Q3 is the transferable result.
- Q8_0 and Q6_K are documented as rejected rather than measured: both were killed after failing to
  produce usable throughput. Their logs are kept in `results/`.
- The first forward pass after loading took 339 s on the Q4 run (cold page cache on a 7.7 GB
  machine). It is excluded from the timed results via an explicit warm-up, and it is the single
  biggest reason a laptop is a poor serving host.
- Ten prompts is a small quality sample; it is enough to catch a broken quantisation, not enough to
  rank two good ones statistically. The checks are objective, which is why they are usable at all
  at this sample size.
