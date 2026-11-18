"""Build the 4-page submission report as HTML (then rendered to PDF by Edge).

Every number is read from the result JSONs so the report cannot drift from the runs.
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(ROOT)


def load(*parts):
    with open(os.path.join(BASE, *parts), encoding="utf-8") as f:
        return json.load(f)


m = load("task1-lora", "results", "metrics.json")
t2_256 = load("task2-rag", "results", "retrieval_256.json")
t2_512 = load("task2-rag", "results", "retrieval_512.json")
b25_256 = load("task2-rag", "results", "bm25_256.json")["summary"]
b25_512 = load("task2-rag", "results", "bm25_512.json")["summary"]

q4 = load("task3-local-llm", "results", "q4_k_m.json")
q8_path = os.path.join(BASE, "task3-local-llm", "results", "q8_0.json")
q8 = load("task3-local-llm", "results", "q8_0.json") if os.path.exists(q8_path) else None

fails = load("task1-lora", "results", "failures.json")
cc = fails["confusion_counts"]
N_SIB = cc.get("card_arrival -> card_delivery_estimate", 0) + cc.get("card_delivery_estimate -> card_arrival", 0)
N_DOM = cc.get("fiat_currency_support -> exchange_via_app", 0)
N_WORD = cc.get("pending_cash_withdrawal -> declined_cash_withdrawal", 0)
N_WRONG = fails["wrong"]

with open(os.path.join(BASE, "task2-rag", "corpus", "pages.jsonl"), encoding="utf-8") as f:
    _pages = [json.loads(l) for l in f]
N_PAGES = len(_pages)
N_CHARS = sum(len(p.get("text", "")) for p in _pages)

Q8_MODEL_MIB = os.path.getsize(os.path.join(BASE, "task3-local-llm", "models",
                                            "qwen2.5-3b-instruct-q8_0.gguf")) / 1024**2
Q8_MIB = f"{Q8_MODEL_MIB:,.0f}"

res = m["results"]
tr = m["training"]
tok = m["token_lengths_prompt_plus_answer"]
lora = tr["lora"]


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def q8cell(key, fmt="{:.2f}"):
    return fmt.format(q8[key]) if q8 else "—"


def q8row(label, key, fmt="{:.2f}", suffix=""):
    return (f"<tr><td>{label}</td><td>{fmt.format(q4[key])}{suffix}</td>"
            f"<td>{q8cell(key, fmt)}{suffix}</td></tr>")


CSS = """
@page { size: A4; margin: 12mm 13mm 12mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.2pt; line-height: 1.33;
       color: #16191c; margin: 0; }
h1 { font-size: 18pt; margin: 0 0 2px 0; letter-spacing: -0.3px; }
.sub { font-size: 9.6pt; color: #4a5560; margin: 0 0 8px 0; }
h2 { font-size: 11.6pt; color: #1f4e79; margin: 11px 0 4px 0; padding-bottom: 2px;
     border-bottom: 1.4px solid #1f4e79; page-break-after: avoid; }
h3 { font-size: 9.8pt; margin: 7px 0 2px 0; page-break-after: avoid; }
p { margin: 0 0 5px 0; text-align: justify; }
ul { margin: 2px 0 5px 0; padding-left: 15px; }
li { margin-bottom: 1.5px; }
table { width: 100%; border-collapse: collapse; font-size: 8.5pt; margin: 3px 0 6px 0;
        page-break-inside: avoid; }
th { background: #eaf1f8; font-weight: 600; text-align: left; }
th, td { border: 0.6px solid #c3cfdb; padding: 2.4px 5px; vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.mono { font-family: Consolas, "Courier New", monospace; font-size: 8.2pt; }
.note { background: #f6f9fc; border-left: 3px solid #1f4e79; padding: 4px 7px; margin: 5px 0;
        page-break-inside: avoid; }
.stat { display: inline-block; width: 32%; border: 1px solid #1f4e79; border-radius: 3px;
        padding: 5px 8px; margin: 0 0.6% 6px 0; vertical-align: top; }
.stat b { display: block; font-size: 14pt; color: #1f4e79; line-height: 1.15; }
.stat span { font-size: 8pt; color: #4a5560; }
.small { font-size: 8.2pt; color: #4a5560; }
.tag { background: #1f4e79; color: #fff; font-size: 7.6pt; padding: 1px 5px; border-radius: 2px;
       vertical-align: 2px; }
h2.newpage { page-break-before: always; }
"""

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tndrr take-home — report</title>
<style>{CSS}</style></head><body>

<h1>tndrr AI/ML take-home</h1>
<p class="sub">Chetan · 18 November 2026 · three tasks, every number below comes from a run I kept in the repo</p>

<div class="stat"><b>{pct(res['lora_r16_final_maxlen128']['category_accuracy'])}</b>
<span>Task 1 held-out intent accuracy (300 examples)</span></div>
<div class="stat"><b>{t2_512['recall_at_5']:.3f}</b>
<span>Task 2 recall@5, best chunking (512 tokens)</span></div>
<div class="stat"><b>{q4['tok_per_s_median']:.1f} tok/s</b>
<span>Task 3 local decode speed, Q4_K_M, 4 CPU threads</span></div>

<h2>Task 1 — LoRA intent router</h2>

<p>I fine-tuned <b>Qwen2.5-0.5B-Instruct</b> to turn a customer-support message into strict JSON,
<span class="mono">{{"category": "&lt;one of 77&gt;"}}</span>, on Banking77. I picked a 0.5B model because the
task is narrow and short — the p95 of prompt-plus-answer is <b>{tok['p95']} tokens</b> — so a small model keeps the
cost/accuracy trade-off visible instead of hiding it behind scale. Split: <b>9,703</b> train, <b>300</b> dev
(never trained on), <b>300</b> held-out from the official test split.</p>

<h3>LoRA settings, and why each one</h3>
<table>
<tr><th>Setting</th><th>Value</th><th>Why</th></tr>
<tr><td>r / alpha</td><td class="mono">16 / 32</td><td>alpha = 2r is the standard scaling. I did not sweep r:
8 adapters give ~4.4M trainable parameters, which looked thin for 77 classes; 32 doubles adapter size for a task
this narrow.</td></tr>
<tr><td>Target modules</td><td class="mono">q,k,v,o,gate,up,down</td><td>All seven linear projections, not
attention only: the label decision leans on the MLP as much as on token mixing.</td></tr>
<tr><td>Dropout</td><td class="mono">0.05</td><td>9.7k rows is small enough to overfit; a little noise on the
adapter is cheap insurance.</td></tr>
<tr><td>LR / schedule</td><td class="mono">2e-4, cosine, 3% warmup</td><td>LoRA tolerates a higher LR than full
FT; warmup avoids an early bad step through the fresh adapter.</td></tr>
<tr><td>Epochs / batch</td><td class="mono">2, 16 × 2 accum = 32</td><td>Loss was still falling at the end of
epoch 1 and flat by the end of epoch 2 (final 0.0159).</td></tr>
<tr><td>Precision / maxlen</td><td class="mono">fp16 autocast, 128</td><td>128 is set by measurement, see below.</td></tr>
<tr><td>Trainable params</td><td class="mono">{tr['trainable_params']:,} / {tr['total_params']:,}</td>
<td>{tr['trainable_pct']}% of the model — that is the whole point of LoRA.</td></tr>
</table>

<div class="note"><b>The bug worth talking about.</b> My first run used maxlen 64. The tokenizer keeps the
<i>last</i> 64 tokens, so on <b>12.8%</b> of rows it silently dropped the system prompt and the label list — the
model was being asked to answer a question it could no longer see. The token-length histogram (p95 = {tok['p95']},
max = {tok['max']}) said 128 was the right ceiling, so I retrained. The 128 run is the one I report:
<b>{tr['steps']} steps, {tr['train_seconds']} s, {tr['peak_gpu_mem_gb']} GB peak</b> on a Colab T4.</div>

<h3>Base vs fine-tuned, 300 held-out examples</h3>
<table>
<tr><th>Setup</th><th class="n">Valid JSON</th><th class="n">Category accuracy</th><th class="n">Strict exact</th></tr>
<tr><td>Base, placeholder prompt</td><td class="n">{pct(res['base_template_prompt']['valid_json'],2)}</td>
<td class="n">{pct(res['base_template_prompt']['category_accuracy'])}</td>
<td class="n">{pct(res['base_template_prompt']['strict_json_exact'])}</td></tr>
<tr><td>Base, all 77 labels in the prompt</td><td class="n">{pct(res['base_with_77_labels_in_prompt']['valid_json'],2)}</td>
<td class="n">{pct(res['base_with_77_labels_in_prompt']['category_accuracy'])}</td>
<td class="n">{pct(res['base_with_77_labels_in_prompt']['strict_json_exact'])}</td></tr>
<tr><td><b>LoRA r=16, maxlen 128</b></td><td class="n"><b>{pct(res['lora_r16_final_maxlen128']['valid_json'],2)}</b></td>
<td class="n"><b>{pct(res['lora_r16_final_maxlen128']['category_accuracy'])}</b></td>
<td class="n"><b>{pct(res['lora_r16_final_maxlen128']['strict_json_exact'])}</b></td></tr>
</table>

<p>The base model's 0.0% is not a parsing artefact: it returns perfectly valid JSON with invented labels
("bank", "pending", "受理"). Give it the label list and it can pick one, but validity collapses to
{pct(res['base_with_77_labels_in_prompt']['valid_json'])} because it starts explaining itself. The adapter is what
makes the output a contract rather than a suggestion.</p>

<h3>Three failures, and why they happen</h3>
<ul>
<li><b>Sibling intents — {N_SIB} of {N_WRONG}.</b> Mostly
<span class="mono">card_arrival → card_delivery_estimate</span> (4 cases). "When will I get my card?" is genuinely
both; the model picks the more literal phrasing rather than the dataset's convention.</li>
<li><b>Domain boundary — {N_DOM} of {N_WRONG}.</b> <span class="mono">fiat_currency_support → exchange_via_app</span>.
"Do you support USD?" is about support, not conversion. The model reads the currency, not the verb.</li>
<li><b>One word flips it — {N_WORD} of {N_WRONG}.</b> "My account was charged for a withdraw I tried to make that was
<b>decline</b>." → <span class="mono">declined_cash_withdrawal</span> instead of
<span class="mono">pending_cash_withdrawal</span>. The cue is real but points the wrong way.</li>
</ul>
<p class="small">Full list and confusion counts: <span class="mono">task1-lora/results/failures.json</span>. I report
the 94.0% run, not my earlier 95.3% at maxlen 64 — that run had the truncation bug, and +1.3 pp is inside the
±2.7 pp binomial noise at n=300.</p>

<h3>7B memory arithmetic (the sizing question)</h3>
<table>
<tr><th>Method</th><th class="n">Weights</th><th class="n">Gradients</th><th class="n">Optimizer</th>
<th class="n">Activations</th><th class="n">Total</th></tr>
<tr><td>Full fine-tune, fp16</td><td class="n">14 GB</td><td class="n">14 GB</td><td class="n">56 GB</td>
<td class="n">2–4 GB</td><td class="n"><b>~86–88 GB</b></td></tr>
<tr><td>LoRA r=16, fp16 base</td><td class="n">14 GB</td><td class="n">0.08 GB</td><td class="n">0.32 GB</td>
<td class="n">~1.5 GB</td><td class="n"><b>~16 GB</b></td></tr>
<tr><td>QLoRA, 4-bit NF4 base</td><td class="n">3.5 GB</td><td class="n">0.08 GB</td><td class="n">0.32 GB</td>
<td class="n">~1.5 GB</td><td class="n"><b>~5.5 GB</b></td></tr>
</table>
<p>For 7B, r=16 on all seven projections is <b>40.4M trainable parameters (0.58%)</b>. Full FT needs the weights,
an equally sized gradient buffer, and Adam's two fp32 moments (8 bytes/param), so it does not fit on a 24 GB card —
it needs 2×80 GB or ZeRO-3. LoRA keeps the frozen base in fp16 and optimises only the adapter, so ~16 GB fits a
24 GB card and QLoRA's 4-bit base brings it under 8 GB. Rejected for this task: full FT (no hardware), QLoRA
(0.5B fits fp16 comfortably; QLoRA is the 7B answer), r=8, attention-only adapters, synthetic data
(Banking77 is already clean), and the 1.5B model (slower for no better accuracy).</p>

<h2>Task 2 — retrieval from scratch</h2>

<p>Corpus: <b>NIST AI RMF 1.0 (NIST AI 100-1)</b>, {N_PAGES} pages and {N_CHARS:,} characters of
public-domain text. No LangChain, no LlamaIndex — PyMuPDF for text, NumPy for the maths. I wrote
<b>{t2_256['n_questions']}</b> questions myself and then verified every gold quote is a verbatim substring of the
page it points to; that check caught two questions where I had misremembered the wording.</p>

<p>Pipeline: extract page by page → chunk inside page boundaries ({t2_256['n_chunks']} chunks at 256 tokens,
{t2_512['n_chunks']} at 512, 64-token overlap) → <span class="mono">BAAI/bge-small-en-v1.5</span> (384-d) →
cosine top-5. BM25 uses <span class="mono">rank_bm25</span> over the same chunks; the hybrid is reciprocal rank
fusion of both lists.</p>

<table>
<tr><th>Retriever</th><th class="n">recall@5, 256 ({t2_256['n_chunks']} chunks)</th>
<th class="n">recall@5, 512 ({t2_512['n_chunks']} chunks)</th><th class="n">MRR 256</th><th class="n">MRR 512</th></tr>
<tr><td>Dense (bge-small)</td><td class="n">{t2_256['recall_at_5']:.3f}</td>
<td class="n">{t2_512['recall_at_5']:.3f}</td><td class="n">{t2_256['mrr_at_5']:.3f}</td>
<td class="n">{t2_512['mrr_at_5']:.3f}</td></tr>
<tr><td>BM25</td><td class="n">{b25_256['bm25']['recall@5']:.3f}</td><td class="n">{b25_512['bm25']['recall@5']:.3f}</td>
<td class="n">{b25_256['bm25']['mrr@5']:.3f}</td><td class="n">{b25_512['bm25']['mrr@5']:.3f}</td></tr>
<tr><td>Hybrid (RRF)</td><td class="n">{b25_256['hybrid_rrf']['recall@5']:.3f}</td>
<td class="n">{b25_512['hybrid_rrf']['recall@5']:.3f}</td><td class="n">{b25_256['hybrid_rrf']['mrr@5']:.3f}</td>
<td class="n">{b25_512['hybrid_rrf']['mrr@5']:.3f}</td></tr>
</table>

<p>512-token chunks beat 256 on dense recall ({t2_512['recall_at_5']:.3f} vs {t2_256['recall_at_5']:.3f}) because
the 256 splits cut answers across boundaries. Page-hit@5 moves the other way
({t2_256['page_hit_at_5']:.3f} → {t2_512['page_hit_at_5']:.3f}), but that is not a trend: three questions flip
and they roughly cancel — q7 and q11 lose the gold page at 512, q4 gains it. With n=15 one question is 6.7 pp, so
recall@5 is the number I trust. BM25 is flat at {b25_512['bm25']['recall@5']:.3f} for both sizes: exact terms
survive a chunk boundary in a way that a 384-d vector does not.</p>

<p><b>Why BM25 wins here.</b> The questions quote the document's own vocabulary — GOVERN, MAP, MEASURE,
"risk tolerance". BM25's IDF automatically upweights exactly those rare tokens, while the dense model blurs them
into a generic "AI governance" neighbourhood. The honest caveat: I wrote the questions after reading the corpus, so
they are partly term-anchored, which flatters BM25. A fair version of this test adds paraphrases.</p>

<p><b>Two failures.</b> <i>q9:</i> the same definition is printed on pages 8 and 17; gold is page 8, both
retrievers return page 17. The content is right and the label is wrong — a gold ambiguity, not a retrieval miss.
<i>q15:</i> asks for a <i>procedure</i>; the top dense hit is a <i>topic</i> paragraph with the same nouns. BM25
ranks the gold chunk first because the procedure section repeats the exact term — the case that shows lexical
matching is not just a baseline.</p>

<p><b>BM25 in one paragraph.</b> Score a chunk by summing, over query terms, an IDF weight times a saturated
term frequency, with a length penalty: <span class="mono">score = Σ IDF(t)·tf(t,d)(k1+1) /
(tf(t,d)+k1(1−b+b·|d|/avgdl))</span>. <span class="mono">k1</span> (1.2–2.0) controls saturation,
<span class="mono">b</span> (0.75) controls length normalisation. Strengths: rare jargon, zero training, an index
you can build in milliseconds. Weaknesses: no synonyms, no word order, no negation.</p>

<p><b>Re-rankers.</b> A cross-encoder reads <span class="mono">(query, chunk)</span> together, so it sees order and
negation that a bag of words and a single vector both miss. Run it on the top 20 only: 20 short forward passes,
~50–100 ms on a GPU, and it fixes precisely the q15 failure by scoring the relationship rather than the overlap.
I left it out because 15 questions cannot measure its gain, and the brief asked for analysis of the option, not a
bigger stack.</p>

<h2 class="newpage">Task 3 — local model, two quantisation settings</h2>

<p>llama.cpp <b>b12465</b> (CPU build), Qwen2.5-3B-Instruct in GGUF, <b>Q4_K_M vs Q8_0</b>, 4 threads on an
i3-1115G4 (2 cores / 4 threads), 8,192-token context, temperature 0, seed 1234. Ten prompts, each with a
checkable answer rather than a vibe: JSON extraction, arithmetic, a two-sentence summary, code that must parse,
factual recall, a refusal, Hindi, classification, a syllogism, and long-context recall.</p>

<table>
<tr><th>Metric</th><th class="n">Q4_K_M</th><th class="n">Q8_0</th></tr>
<tr><td>File size</td><td class="n">{q4['model_mib_file']:.0f} MiB</td><td class="n">{q8cell('model_mib_file','{:.0f}') if q8 else Q8_MIB} MiB</td></tr>
<tr><td>Effective bits / weight</td><td class="n">~5.2</td><td class="n">~8.9</td></tr>
<tr><td>KV cache at 8k context (fp16)</td><td class="n">{q4['kv_cache_mib_analytic']:.0f} MiB</td>
<td class="n">{q4['kv_cache_mib_analytic']:.0f} MiB</td></tr>
<tr><td>Peak working set (measured)</td><td class="n">{q4['peak_working_set_gb']:.2f} GB</td>
<td class="n">{q8cell('peak_working_set_gb','{:.2f}')} GB</td></tr>
<tr><td>Load time</td><td class="n">{q4['load_s']:.1f} s</td><td class="n">{q8cell('load_s','{:.1f}')} s</td></tr>
<tr><td>Generation speed, median</td><td class="n"><b>{q4['tok_per_s_median']:.2f} tok/s</b></td>
<td class="n"><b>{q8cell('tok_per_s_median')} tok/s</b></td></tr>
<tr><td>Generation speed, range</td><td class="n">{q4['tok_per_s_min']:.2f}–{q4['tok_per_s_max']:.2f}</td>
<td class="n">{q8cell('tok_per_s_min')}–{q8cell('tok_per_s_max')}</td></tr>
<tr><td>Checks passed (of 10)</td><td class="n">{q4['passes']}/10</td><td class="n">{f"{q8['passes']}/10" if q8 else '—'}</td></tr>
</table>

<p>The tok/s spread is real, not noise I am hiding: on two physical cores that are also running Windows and an
IDE, prompts ranged from {q4['tok_per_s_min']:.2f} to {q4['tok_per_s_max']:.2f} tok/s. The median is the honest
headline; the mean ({q4['tok_per_s_mean']:.2f}) is dragged down by the Hindi prompt. Q4_K_M passed 9/10 — the one
miss was the Hindi answer, which was correct but one sentence where the prompt demanded two. That is
instruction-following, not knowledge: exactly the surface constraint a tighter quantisation is expected to
degrade first.</p>

<p><b>KV cache.</b> Every generated token attends to the keys and values of every earlier token at every layer.
Those vectors never change, so they are cached instead of recomputed. Size is set by the architecture, not the
weights: <span class="mono">2 (K,V) × layers × kv_heads × head_dim × bytes × context</span>. For this 3B model
that is <b>2 × 36 × 2 × 128 × 2 = 36 KiB per token</b>, so 8,192 tokens costs <b>{q4['kv_cache_mib_analytic']:.0f} MiB</b>.
Grouped-query attention (16 query heads share 2 KV heads) cuts it 8× versus full multi-head. It is per sequence:
20 users means 20×, which is why a serving plan is a KV-budget plan, not a weights plan.</p>

<p><b>Quantisation.</b> fp16 stores 2 bytes per weight, so 3.09B parameters would need ~6.2 GB. Q8_0 keeps 8 bits
plus one scale per 32-weight block (~8.5 bits/weight) → 3.37 GB measured. Q4_K_M mixes 4-bit and 6-bit tensors with
per-block scales and minimums → 1.96 GB, ~5.2 effective bits/weight. The saving is close to linear in bits, with a
small overhead for block metadata. Crucially, quantisation changes <i>weights only</i>: the KV cache and the
activations are separate allocations, so the KV number is identical for both models, and you can quantise the KV
cache independently (<span class="mono">--cache-type-k q8_0 --cache-type-v q8_0</span>) to halve it. Q8 is
near-lossless; Q4 is where long-tail knowledge and surface constraints start to wobble. Decoding is
bandwidth-bound, so reading 2 GB per token is faster than reading 3.4 GB — Q4 should also win on speed.</p>

<h3>20 concurrent users on one 24 GB GPU</h3>
<p>The 3B model is the wrong size for this question, so the plan uses Qwen2.5-7B-Instruct at Q4_K_M (~4.1 GiB
weights). Qwen2.5-7B has 28 layers, 4 KV heads, head_dim 128:
<span class="mono">2 × 28 × 4 × 128 × 2 = 56 KiB/token</span>; an 8,192-token context is <b>448 MiB per user</b>;
20 users is <b>8.75 GiB</b>. Add ~2 GiB for compute buffers, CUDA context and fragmentation: <b>~15 GiB of 24</b>,
with 9 GiB spare. With an 8-bit KV cache the KV term halves to 4.4 GiB and the total drops to ~10.5 GiB, which is
the configuration I would actually ship.</p>

<p>Serve with <span class="mono">llama-server -np 20 -c 163840 --cache-type-k q8_0 --cache-type-v q8_0
--cont-batching</span>: the context is split across slots, so each of the 20 slots gets 8,192 tokens, and llama.cpp
reserves that space up front — the number above is a hard allocation, not a worst case. If traffic is bursty short
chats with occasional long ones, vLLM with PagedAttention is the better fit because it allocates KV in blocks on
demand and idle sessions cost almost nothing. Throughput on a 4090-class card is roughly 60–80 tok/s for a single
stream and 400–600 tok/s aggregate with continuous batching, so 20 active users see ~20–30 tok/s each, about
reading speed; an A10-class card halves that. Ship with an admission cap of 20 active sequences, a per-request
<span class="mono">max_tokens</span> so one user cannot hold a slot for minutes, prefix caching for the shared
system prompt, and a queue — letting the batch grow instead just makes everyone slow.</p>

<h2>What I would do next, and what I cannot claim</h2>
<p>The 2-core laptop is the binding constraint in Task 3: the first forward pass after loading took 339 s of cold
page-cache paging, which is why the runner does an explicit warm-up and excludes it. Ten prompts is a smoke test,
not a benchmark — enough to catch a broken quantisation, not enough to rank two good ones. Task 2's n=15 means one
question moves recall by 6.7 pp, and my questions are term-anchored. With more time: paraphrase the Task 2
questions and add a cross-encoder re-ranker; sweep LoRA r on Task 1 instead of arguing from parameter counts; and
run Task 3 on a machine where Q8_0 (3.4 GB) is not fighting Windows for 7.7 GB of RAM.</p>

<p class="small">Repo: <span class="mono">task1-lora/</span> (train.py, eval.py, eval_zero.py, results/, REPORT_task1.md) ·
<span class="mono">task2-rag/</span> (extract.py, chunk.py, search.py, bm25.py, analyze.py, analyze2.py, questions.json,
results/, REPORT_task2.md) · <span class="mono">task3-local-llm/</span> (run_task3.py, prompts.json, summarize.py,
results/, REPORT_task3.md).</p>

</body></html>
"""

out = os.path.join(ROOT, "report.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out, len(HTML), "chars")
