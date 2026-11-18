# tndrr AI/ML internship — assignment

Submission for the tndrr AI/ML internship assignment (deadline 19 Nov 2026).

| Folder | Task | Status |
|---|---|---|
| [`task1-lora/`](task1-lora/) | LoRA fine-tuning of a ≤1.5B model | done — code, metrics, failures, write-up |
| [`task2-rag/`](task2-rag/) | RAG from scratch (no LangChain/LlamaIndex) | done — code, 15 questions, recall@5, failures |
| [`task3-local-llm/`](task3-local-llm/) | Local model, two quantisation settings | done — llama.cpp, Q4_K_M vs Q3_K_M, 10 checks |

## task1-lora at a glance

Qwen2.5-0.5B-Instruct + LoRA (r=16, 1.75% of params) on Banking77, output constrained to
`{"category": "<one of 77 intents>"}`. Trained on a free T4 in 436 s.

| | Base | + LoRA |
|---|---:|---:|
| Held-out category accuracy (n=300) | 0.0% | **94.0%** |
| Valid JSON | 99.7% | **100%** |

See [`task1-lora/README.md`](task1-lora/README.md) for the full method, settings rationale,
failure analysis and the 7B memory calculation.

## task2-rag at a glance

NIST AI RMF 1.0 (48 pages) → page-bounded token chunks → `bge-small-en-v1.5` → cosine in NumPy.
15 hand-written questions with verbatim gold passages, all machine-verified against their page.

| Retriever | 256-token chunks | 512-token chunks |
|---|---:|---:|
| Dense recall@5 | 0.667 | **0.733** |
| BM25 recall@5 | 0.800 | 0.800 |

512 works better for embeddings because 256 splits answer sentences across chunk boundaries: the
right page is found 80% of the time but the right chunk only 66.7%. Two failures are analysed
(a duplicate-content gold-label problem, and topic beating procedure), plus measured BM25 and
RRF hybrid comparisons.

See [`task2-rag/README.md`](task2-rag/README.md) for method, per-question results and the
BM25 / re-ranker discussion.

## task3-local-llm at a glance

Qwen2.5-3B-Instruct GGUF under llama.cpp b12465, CPU-only (4 threads), 8k context, ten
prompts with programmatic checks.

| | Q4_K_M | Q3_K_M |
|---|---:|---:|
| File size | 2,007 MiB | 1,644 MiB |
| Decode speed (median) | 9.12 tok/s | 11.46 tok/s |
| Peak RSS | 1.86 GB | 1.54 GB |
| Checks passed | 9/10 | 8/10 |

The one Q4 miss is the Hindi prompt: correct language, one sentence where the prompt demanded
two. I planned Q4_K_M vs Q8_0, but Q8_0 (3.37 GiB) never completed a forward pass on this 7.7 GB
laptop, and Q6_K (2.60 GiB) managed 0.06 tok/s — 17.5 s per token. Both were paging, not
computing. The ceiling sits between Q4_K_M and Q6_K, so the second setting is Q3_K_M (1.61 GiB),
the next rung down that fits: it runs 1.26× faster and 1.20× smaller in peak RSS, at the cost of
one extra check (the two-sentence constraint). Full analysis — KV cache maths and a 20-user /
24 GB serving plan — in [`task3-local-llm/REPORT_task3.md`](task3-local-llm/REPORT_task3.md).

## Reproducing

Task 1 needs a GPU (free Colab T4 is enough, ~7 min end to end). Task 2 runs on CPU (~20 s of
embedding). Task 3 needs llama.cpp b12465 (binaries not committed) plus ~3.6 GiB of GGUF files;
the commands are in `task3-local-llm/README.md`.
