# Task 2 — Retrieval from scratch (no LangChain / LlamaIndex)

**Corpus:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1) — 48 pages, 105,949 characters, public domain
**Embedder:** `BAAI/bge-small-en-v1.5` (384-dim), cosine similarity in NumPy
**Question set:** 15 questions I wrote, each with a verbatim gold passage and page number
**Result:** dense recall@5 **0.667** at 256 tokens vs **0.733** at 512 tokens

---

## 1. Setup

| Piece | Choice | Why |
|---|---|---|
| Corpus | NIST AI RMF 1.0 | Real, public, 48 pages, proper text layer, and it repeats content across sections — which is a fair test rather than a clean toy |
| Extraction | PyMuPDF → per-page text, whitespace-normalised | Keeps page numbers, so every retrieved chunk is citable |
| Chunker | token-based, **page-bounded** | A chunk that spans two pages cannot be cited by page number |
| Embeddings | `bge-small-en-v1.5`, normalised | 384-dim, runs on CPU, 512-token window |
| Index | NumPy dot product | Both sides normalised, so dot = cosine |
| Metric | recall@5 (and MRR@5) | "Is the passage containing the answer in the top 5" |

Chunks are sliced on the original characters through the tokenizer's offset mapping, so the
stored text is byte-identical to the source. That matters: gold matching is exact substring
containment, not fuzzy.

## 2. Chunking

| | 256 tokens | 512 tokens |
|---|---:|---:|
| Chunks | 116 | 62 |
| Median chunk length | 256 | 408 |
| Overlap | 64 tokens | 64 tokens |

512 is the upper bound because `bge-small` has a 512-token window; a 512-token chunk plus the
`[CLS]`/`[SEP]` specials truncates by at most two tokens, which the overlap already covers.

## 3. Results (n = 15, 1 question = 6.7 pp)

| Retriever | Chunk 256 | Chunk 512 |
|---|---:|---:|
| **Dense (bge-small + NumPy), recall@5** | **0.667** | **0.733** |
| Dense, MRR@5 | 0.394 | 0.422 |
| Dense, gold page in top-5 | 0.800 | 0.733 |
| BM25, recall@5 | 0.800 | 0.800 |
| BM25, MRR@5 | 0.619 | 0.589 |
| Hybrid (dense + BM25, RRF), recall@5 | 0.800 | 0.800 |

### Which chunk size worked better, and why

**512, for the dense retriever** (0.733 vs 0.667). The reason is visible in the page-hit row:
at 256 the correct *page* is in the top 5 80% of the time but the correct *chunk* only 66.7% of
the time (0.800 vs 0.667). At 512 those two numbers are identical (0.733 / 0.733). In other
words, the 256-token retriever usually finds the right page and then hands back the neighbouring
chunk — the answer sentence was split across a boundary. Two questions (`q10`, `q13`) hit the page
but not the chunk at 256; at 512 that count is zero.

A 512-token chunk holds a complete subcategory statement ("GOVERN 1.5: … including determining
the frequency of periodic review."), so its vector represents one whole proposition instead of
the front half of one.

**But the honest caveat:** for BM25 the difference disappears (0.800 both), and 256 is slightly
better on MRR (0.619 vs 0.589), because shorter chunks concentrate the term matches instead of
diluting them with surrounding prose. So "which size is better" depends on the retriever — 512 is
better for embeddings, and roughly a wash for lexical search.

## 4. Two retrieval failures, and what would fix them

### Failure A — the right content, the wrong location (q9)

> **Q:** *Which characteristics of trustworthy AI are listed in the Executive Summary?*
> **Gold (p8):** "valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy enhanced, and fair with their harmful biases managed"

Both chunk sizes put **p17** ("3. AI Risks and Trustworthiness") at rank 1, and p5 at ranks 2–3.
BM25 misses too. The reason is that this framework states the same characteristic list twice —
once in the Executive Summary and once in §3 — so the retriever returns a semantically correct
passage that is not the page the gold label points at.

It is worth being blunt about what this is: **a gold-label problem as much as a retrieval
problem.** A top-5 that contains p17 has found the facts; strict chunk-containment still scores
it zero. Fixes: accept either location in the gold set, collapse near-duplicate passages at index
time, or make the label section-aware (the question says "Executive Summary", and no retriever
here models document structure).

### Failure B — topic beats procedure (q15)

> **Q:** *What is required for regularly engaging relevant AI actors and integrating their feedback?*
> **Gold (p33):** "MAP 5.2: Practices and personnel for supporting regular engagement with relevant AI actors and integrating feedback about positive, negative, and unanticipated impacts are in place and documented."

Dense returns p40, p14, p46, p41, p45 — all pages that discuss **AI actors** as a topic. The
embedding matched the dominant subject of the query, not the procedural requirement. The gold
sentence contains the query almost verbatim ("regular engagement … AI actors … integrating
feedback"), and **BM25 ranks it 1st**, which is exactly the case where lexical matching beats a
single averaged vector.

Fix: hybrid retrieval (BM25 recovered it here — see the table), or a cross-encoder re-ranker over
the top ~50, which can compare the query against the sentence's function ("is required") rather
than its topic. Query rewriting ("what does MAP 5.2 require") would also do it, but that is
tuning the test to the answer.

## 5. What BM25 adds to embedding search

BM25 scores a passage by exact term overlap, weighted by how rare each term is in the corpus
(IDF), with term-frequency saturation and a document-length correction. Embeddings do the
opposite: they compress the passage into one 384-dimension vector, which is what gives them
synonym and paraphrase ability but also what averages away a single rare token.

So BM25 adds **precision on exact strings** — identifiers, section numbers ("GOVERN 1.5",
"MEASURE 2.6"), product names, numbers, and unusual phrasing. In this corpus that is most of what
makes a passage the *right* passage rather than a topically adjacent one. Embeddings add the
opposite: they find "how do I stop my card being frozen" ≈ "account suspension", which BM25 cannot
do at all because no words are shared.

Measured here: BM25 alone beat dense at both chunk sizes (0.800 vs 0.667/0.733). Fusing the two
with reciprocal rank fusion reached 0.800 as well, and no single method retrieved everything —
`q4` and `q12` are missed by BM25 while `q15` is missed by dense, which is the argument for a
hybrid in one line.

## 6. What a re-ranker does

A re-ranker is a second-stage scorer. A first-stage retriever (dense or BM25) returns, say, the
top 50 candidates cheaply. The re-ranker then takes each `(query, passage)` pair and runs a
**cross-encoder** — a model that reads both texts together, so every query token can attend to
every passage token. That joint attention is what lets it tell "what MAP 5.2 requires" apart from
"general discussion of AI actors", which is precisely the q15 failure.

The trade-off is cost: one forward pass per candidate pair, so it cannot run over the whole
corpus, only over the shortlist. It improves **ordering** (MRR, nDCG) much more than **recall** —
it can reorder what stage 1 found, but it cannot recover a passage stage 1 never returned. In my
results the natural place for it is the MRR column (0.39–0.42), which is where ordering losses live.

## 7. Honest limitations

- **n = 15.** One question is 6.7 pp. The 0.733 vs 0.667 gap is exactly one question, so I do not
  claim it is statistically solid — I claim the *mechanism* is visible in the page-hit numbers.
- **My questions are partly term-anchored.** I wrote them from the document's vocabulary, which
  favours BM25. A paraphrase-only set would move the balance toward dense. This is the single
  biggest caveat on the BM25 result.
- **One corpus.** A procedural standards document is unusually friendly to lexical matching.
- No semantic chunking, no metadata filters, no query expansion — all of which would raise these
  numbers, and all of which I left out to keep the pipeline something I can defend line by line.

## 8. Reproduce

```bash
pip install sentence-transformers rank-bm25 pymupdf

python extract.py          # PDF -> corpus/pages.jsonl
python verify_questions.py # checks all 15 gold quotes are verbatim on the cited page
python chunk.py            # -> corpus/chunks_256.jsonl, chunks_512.jsonl
python search.py 256       # dense retrieval, writes results/retrieval_256.json
python search.py 512
python bm25.py             # BM25 + RRF hybrid, writes results/bm25_<size>.json
python analyze.py          # per-question comparison
```

Files: `questions.json` (15 Q + gold passage + page), `results/retrieval_*.json` (top-5 per
question, with scores), `results/bm25_*.json` (dense vs BM25 vs hybrid).
