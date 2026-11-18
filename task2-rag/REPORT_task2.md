# Task 2 — Retrieval (RAG) from scratch

Built without LangChain or LlamaIndex: PyMuPDF for extraction, a tokenizer-based chunker,
`bge-small-en-v1.5` embeddings, and cosine similarity in NumPy. BM25 was added as a comparison.

**Corpus:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1), 48 pages / 105,949 characters,
public domain. It was chosen because a standards document repeats the same content in several
sections — a fair test, not a clean toy.

**Questions:** 15 written by me, each with a verbatim gold passage and page number. All 15 gold
quotes are verified programmatically to occur on the page they cite (`verify_questions.py`).

## Retrieval accuracy

| Retriever | Chunk 256 (116 chunks) | Chunk 512 (62 chunks) |
|---|---:|---:|
| **Dense (bge-small + NumPy) — recall@5** | **0.667** | **0.733** |
| Dense — MRR@5 | 0.394 | 0.422 |
| Dense — gold page anywhere in top-5 | 0.800 | 0.733 |
| BM25 — recall@5 | 0.800 | 0.800 |
| Hybrid (dense + BM25, RRF) — recall@5 | 0.800 | 0.800 |

**512 tokens worked better for the embedding retriever.** The page-hit row explains why: at 256
the correct page is retrieved 80% of the time but the exact chunk only 66.7% — so it finds the
right page and returns the neighbouring chunk, because the answer sentence straddles a chunk
boundary. At 512 the two numbers are identical (0.733), meaning whenever the right page is found
the right chunk is in the same window. A 512-token chunk holds one complete subcategory statement
and its vector represents a whole proposition rather than half of one.

For BM25 the chunk size makes no difference to recall (0.800 both), and 256 is marginally better on
MRR (0.619 vs 0.589) because shorter chunks concentrate term matches. So the answer is
retriever-dependent: 512 for dense, a wash for lexical.

**The main caveat:** with n = 15, one question is 6.7 pp, and 0.733 vs 0.667 is exactly one
question. The mechanism is visible in the page-hit numbers, but the margin is not statistically
strong, and I say so rather than overclaiming.

## Two failures

**A. Right content, wrong location (q9).** "Which characteristics of trustworthy AI are listed in
the Executive Summary?" → gold on p8, but both chunk sizes put p17 at rank 1 and BM25 misses as
well. The framework states the same characteristic list twice, so the retriever returns a
semantically correct passage that is not the labelled one. This is as much a gold-label problem as
a retrieval problem: a top-5 containing p17 has found the facts. Fixes: accept either location as
gold, deduplicate near-identical passages at index time, or make labels section-aware.

**B. Topic beats procedure (q15).** "What is required for regularly engaging relevant AI actors and
integrating their feedback?" → dense returns p40/p14/p46 (pages about AI actors as a topic) and
misses the MAP 5.2 requirement on p33. The embedding matched the query's subject, not its
function. The gold sentence contains the query almost verbatim, and **BM25 ranks it 1st** — the
textbook hybrid case. Fix: fuse BM25 with dense, or add a cross-encoder re-ranker that reads the
query and passage jointly.

## What BM25 adds to embedding search

BM25 scores exact term overlap, weighted by term rarity (IDF), with term-frequency saturation and a
length correction. An embedding compresses a passage into one vector, which buys paraphrase and
synonym matching but averages away any single rare token. So BM25 adds **precision on exact
strings** — identifiers, section numbers ("GOVERN 1.5", "MEASURE 2.6"), numbers, unusual phrasing —
which in a standards document is often what distinguishes the right passage from a topically
adjacent one. Embeddings do the opposite job: they match "my card is frozen" to "account
suspension". Measured here, BM25 beat dense at both chunk sizes and no single method retrieved
everything (q4 and q12 are missed by BM25, q15 by dense), which is the argument for a hybrid.

## What a re-ranker does

A re-ranker is a second-stage cross-encoder: for each `(query, passage)` pair from the first-stage
shortlist (say top 50), it reads both texts together so every query token can attend to every
passage token, then scores relevance. That joint reading is what separates "what does MAP 5.2
require" from "a general discussion of AI actors" — the q15 failure. The cost is one forward pass
per pair, so it only runs over the shortlist, never the corpus. It mainly improves **ordering**
(MRR, nDCG), not **recall**: it can reorder what stage 1 returned but cannot recover a passage
stage 1 never found.
