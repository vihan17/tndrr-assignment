# Task 1 — LoRA fine-tune: Banking77 → strict JSON intent routing

**Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (502.8M params)
**Task:** a plain-English bank support ticket → exactly `{"category": "<one of 77 intents>"}`
**Hardware:** free Colab Tesla T4 (15.6 GB), fp16 autocast
**Data:** [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) — 10,003 train / 3,080 test, 77 classes
**Result:** held-out accuracy **0.0% → 94.0%**, valid JSON **99.7% → 100%**, cost **436 s** and **9.12 GB** peak

---

## 1. Why this task

Banking77 is a real public dataset of customer support queries labelled with 77 intents. It
gives a narrow, closed output space (good for a 0.5B model) but classes that are genuinely
confusable (`card_arrival` vs `card_delivery_estimate`, several `*_fee_charged` variants), so
the evaluation is not trivial. The output is a fixed JSON object, which is exactly the
"plain-English request → fixed JSON" shape the assignment asks for.

## 2. Data splits

| Split | Size | Use |
|---|---|---|
| Train | 9,703 | gradient updates |
| Dev | 300 (carved from train, seed 0) | monitoring during training only |
| Held-out test | 300 (first 300 of the official test split) | reported numbers, never trained on |

## 3. Metrics

All three are computed on greedy-decoded output with the same prompt the model was trained on:

- **`valid_json`** — output parses as JSON and contains a string `category` field.
- **`category_accuracy`** — parsed `category` equals the gold label (77-way, chance = 1.3%).
- **`strict_json_exact`** — the raw string is byte-identical to the gold target
  `{"category":"<label>"}` (no spaces, no prose around it).

## 4. Results (held-out test, n = 300)

| Model | Valid JSON | Category accuracy | Strict exact |
|---|---:|---:|---:|
| Base, prompt with a `<label>` placeholder | 99.7% | **0.0%** | 0.0% |
| Base, all 77 labels listed in the prompt | 63.7% | **21.0%** | 0.0% |
| **LoRA r=16 (final, maxlen 128)** | **100%** | **94.0%** | **94.0%** |

Two things worth noting:

- The base model is **not** format-limited. With the placeholder prompt it emits valid JSON
  99.7% of the time — it just has no idea what the label vocabulary is, and fills the slot with
  generic words (`"bank"`, `"pending"`, even Chinese `"受理"`). Adding the 77 labels to the
  prompt only gets to 21% and *hurts* JSON validity (long prompt, model starts talking).
  So the LoRA gain is task knowledge, not formatting ability.
- An identical earlier run with `maxlen=64` scored **95.3%** on the same 300 examples. That
  ~1.3 pp gap between two runs of the same recipe is run-to-run noise at n=300 — worth stating
  rather than picking the nicer number.

Dev set (300 examples excluded from training): 0.0% → **90.3%** accuracy, valid JSON 1.0.

## 5. LoRA settings and why

| Setting | Value | Why |
|---|---|---|
| `r` | 16 | The task needs new *label semantics*, not just a new style. r=8 is usually enough for format adaptation; 77 overlapping classes needed more capacity. Still only 1.75% of params. |
| `lora_alpha` | 32 | α = 2r, the common scaling convention; effective adapter scale α/r = 2 without touching the LR. |
| `lora_dropout` | 0.05 | 9.7k rows × 2 epochs is small; light regularisation, not enough to slow convergence. |
| `target_modules` | q,k,v,o + gate,up,down | All linear layers. Attention-only adapters learn format well but struggle when the mapping to the answer space lives in the MLP blocks. Cost is still tiny (8.8M params). |
| `learning_rate` | 2e-4 | LoRA tolerates higher LR than full FT; 1e-3 was the risky end at effective batch 32, 1e-4 would need more epochs. |
| `epochs` | 2 | Loss reached 0.016 and dev accuracy 90%; a third epoch on 9.7k rows mostly memorises. |
| batch | 16 × grad-accum 2 | Fits the T4 with headroom (9.12 GB peak); effective 32 stabilises LoRA gradients. |
| `max_seq_len` | 128 | Measured prompt+answer lengths: p50 57, p95 72, p99 88, max 123. A 64-token cap silently truncated the *system prompt* in 12.8% of rows (prompt is right-truncated, label is never truncated) — a train/inference mismatch, so I raised it. |

Loss is computed on the answer tokens only; the prompt is masked with `-100`.

## 6. Three failures and why (18 / 300 wrong)

The errors are not random — they cluster into three families.

**1. `card_arrival` → `card_delivery_estimate` (4 of 18 errors)**
> *"How do I know when my card will arrive?"* → predicted `card_delivery_estimate`, gold `card_arrival`

Both intents are about card timing. The real distinction is a state the text does not carry —
whether the card already exists on the account (arrival) or has not been sent yet (delivery
estimate). The phrase *"will arrive"* is a strong lexical cue for `delivery_estimate`, and
~126 training examples per class are not enough to override it.

**2. `fiat_currency_support` → `exchange_via_app` (3 of 18 errors)**
> *"Do you offer additional currency options?"* → predicted `exchange_via_app`, gold `fiat_currency_support`

`fiat_currency_support` is a policy question ("which currencies do you support"). Its surface
form overlaps heavily with `exchange_via_app` ("currencies", "exchange", "options"), and there
are several currency-adjacent classes in the 77. The model learned word-level cues instead of
the question type, so it splits the currency family by whichever noun appears.

**3. `pending_cash_withdrawal` → `declined_cash_withdrawal`**
> *"My account was charged for a withdraw I tried to make that was decline."* → predicted `declined_cash_withdrawal`

A single token ("decline") flips the decision. The two classes share the same frame and differ
in transaction outcome, which is not in the query text. This is the same failure mode as #1:
one dominant cue beats the actual intent.

**What would fix them:** (a) targeted contrastive examples — minimal pairs that differ only in
the distinguishing word; (b) putting short label descriptions in the system prompt so the
distinction is stated rather than inferred; (c) kNN retrieval of the nearest labelled training
examples as few-shot context. A fourth, smaller factor: some of the 77 labels are themselves
arguable (`extra_charge_on_statement` vs the `*_fee_charged` family), so a few of the 18 are
boundary cases rather than model errors.

## 7. Memory: 7B with (a) full FT, (b) LoRA, (c) QLoRA

Assumptions: 7e9 params, AdamW, mixed precision (fp16/bf16 compute, fp32 master weights),
activations quoted per batch=1 × 2048 tokens.

**(a) Full fine-tuning**

| Item | Bytes/param | Size |
|---|---|---:|
| fp16 weights | 2 | 14 GB |
| fp16 gradients | 2 | 14 GB |
| fp32 master weights | 4 | 28 GB |
| Adam m | 4 | 28 GB |
| Adam v | 4 | 28 GB |
| **states subtotal** | **16** | **112 GB** |
| activations (grad checkpointing) | — | ~2–4 GB |

→ **≈115 GB minimum.** A single 80 GB A100 does not fit; you need at least 2×80 GB with
FSDP/ZeRO-3, and 4×80 GB in practice.

**(b) LoRA** (frozen fp16 base, r=16 on the 7 linear projections)

Adapter size per layer (d=4096, ffn=11008):
`4 × (16×4096 + 16×4096) + 3 × (16×4096 + 16×11008) = 524,288 + 724,992 = 1,249,280`
× 32 layers ≈ **40M params (0.57% of 7B)**.

| Item | Size |
|---|---:|
| fp16 base weights (frozen, no grads) | 14 GB |
| adapter weights + grads + fp32 Adam (40M × 16 B) | 0.64 GB |
| activations (checkpointing) | ~1–3 GB |

→ **≈16–18 GB** → fits one 24 GB card (RTX 4090 / L4 / A10).

**(c) QLoRA** (NF4 4-bit base + the same adapter)

| Item | Size |
|---|---:|
| 4-bit base weights (7e9 × 0.5 B) + quant constants | ~3.6 GB |
| adapter weights + grads + paged Adam | ~0.6 GB |
| activations | ~1–2 GB |

→ **≈5.5–7 GB** → fits a 12 GB card, comfortably on a 16 GB T4.

**Rule of thumb:** full FT ≈ 16 B/param, LoRA ≈ 2 B/param for the frozen base plus a
negligible adapter, QLoRA ≈ 0.5 B/param. The measured 0.5B LoRA run peaked at 9.12 GB, which
is mostly activations — weights were only ~1 GB — the same reason (b) and (c) are dominated by
activation memory once you raise batch size.

## 8. Reproduce

```bash
pip install transformers peft datasets accelerate
pip uninstall -y torchao      # Colab ships torchao 0.10; peft 0.21 raises ImportError on it

python train.py --epochs 2 --maxlen 128 --out runs/lora_r16
python eval.py --n 300 --out results/base
python eval.py --adapter runs/lora_r16 --n 300 --out results/lora
python eval_zero.py --n 300 --out results/zero
```

`results/metrics.json` holds the summary numbers, `results/failures.json` the 18 error cases.
`eval.py` writes the full `predictions.jsonl` (300 rows) so every number above can be re-derived.

**Retained artifacts:** the metrics and failure JSONs are committed; the adapter (`runs/lora_r16/`)
and `predictions.jsonl` are not — the Colab runtime was recycled after the run. The commands above
regenerate both from the fixed seed 0 in about 7 minutes on a free T4.

## 9. Limitations

- One seed, one run per configuration; ±1–2 pp variance at n=300 is expected.
- 300 test examples is enough to rank these three settings clearly, not enough for fine-grained
  comparisons between LoRA ranks.
- The 77-way accuracy is not comparable to published banking77 numbers, which use the full
  3,080-example test set and often larger models.
