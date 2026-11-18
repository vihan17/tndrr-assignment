# Task 1 — Fine-tuning with LoRA

## Setup

Base model `Qwen/Qwen2.5-0.5B-Instruct`, fine-tuned on Banking77 (10,003 train / 3,080 test,
77 intents) to turn a support ticket into exactly `{"category": "<intent>"}`. All runs were on a
free Colab Tesla T4 (15.6 GB) in fp16, with a hand-written loop over `transformers` + `peft`
(no TRL). 9,703 rows were used for training, 300 held out as a dev set, and the reported numbers
are on 300 examples from the official test split that were never trained on.

## Results (held-out test, n = 300)

| Model | Valid JSON | Category accuracy | Byte-exact JSON |
|---|---:|---:|---:|
| Base, placeholder prompt | 99.7% | 0.0% | 0.0% |
| Base, all 77 labels in prompt | 63.7% | 21.0% | 0.0% |
| **LoRA r=16** | **100%** | **94.0%** | **94.0%** |

The base model formats JSON correctly (99.7%) but has no label vocabulary — it fills the slot
with words like `"bank"` or `"pending"`. Giving it the 77 labels in the prompt only reaches 21%
and *lowers* JSON validity to 63.7%. So fine-tuning adds task knowledge, not formatting ability.
A repeat run with a 64-token cap scored 95.3% on the same set; that ±1.3 pp is noise at n=300,
and I report the 128-token run because a shorter cap truncated the system prompt in 12.8% of
training rows (measured: p95 = 72 tokens).

## LoRA settings

`r=16, alpha=32, dropout=0.05`, on **all seven** linear projections (`q,k,v,o,gate,up,down`),
LR `2e-4` cosine with 3% warmup, 2 epochs, effective batch 32 (16 × grad-accum 2). Trainable
parameters: **8.80M / 502.8M = 1.75%**. Training took 436 s and peaked at 9.12 GB.

- **r=16**: r=8 is normally enough for format adaptation; this task needs new label *semantics*
  across 77 overlapping classes, so I doubled capacity. It is still 1.75% of the model.
- **alpha=32 = 2r**: the standard scaling convention, giving effective scale α/r = 2 without
  changing the LR.
- **All linear layers**: attention-only adapters learn output style but not mappings that live
  in the MLP blocks.
- **2 epochs**: loss reached 0.016 and dev accuracy 90%; a third epoch on 9.7k rows mostly
  memorises.

## Three remaining failures (18 / 300 wrong)

1. **"How do I know when my card will arrive?"** → `card_delivery_estimate`, gold `card_arrival`.
   Four of the 18 errors are this pair. Both are card-timing intents; the real difference is a
   state the text does not contain (card already on the account vs not yet sent). "Will arrive"
   is a strong lexical cue, and ~126 examples per class cannot override it.
2. **"Do you offer additional currency options?"** → `exchange_via_app`, gold `fiat_currency_support`.
   Three errors are this pair. `fiat_currency_support` is a policy question with no distinctive
   surface form; it shares "currency/exchange/options" with `exchange_via_app`. The model keyed
   on words rather than on the question type.
3. **"My account was charged for a withdraw I tried to make that was decline."** →
   `declined_cash_withdrawal`, gold `pending_cash_withdrawal`. One token ("decline") flips the
   answer; the gold distinction is transaction state, absent from the text.

All three are the same failure mode: a single dominant word beats the underlying intent, because
9.7k rows × 2 epochs only teach surface cues. Fixes: minimal-pair contrastive examples, short
label descriptions in the system prompt, or kNN retrieval of labelled neighbours as few-shot
context. A few borderline cases also reflect arguable gold labels inside the `*_fee_charged` family.

## Memory for a 7B model

Assumptions: 7e9 params, AdamW, mixed precision (fp16 weights/compute + fp32 master weights),
activations for batch 1 × 2048 tokens.

**(a) Full fine-tuning** — weights 14 GB (2 B/param) + gradients 14 GB (2 B) + fp32 master 28 GB
(4 B) + Adam m 28 GB + Adam v 28 GB = **112 GB** of state alone, plus 2–4 GB of activations with
checkpointing. Total **≈115 GB**: a single 80 GB card does not fit; 2×80 GB with FSDP/ZeRO-3 is
the minimum, 4×80 GB realistic.

**(b) LoRA** — the fp16 base is frozen, so it is 14 GB with no gradients and no optimizer state.
The adapter is
`4 × (16·4096 + 16·4096) + 3 × (16·4096 + 16·11008) = 1,249,280` per layer, × 32 layers ≈ 40M
params (0.57%), and at 16 B/param (weights + grads + Adam) that is 0.64 GB. With 1–3 GB of
activations: **≈16–18 GB** — one 24 GB card.

**(c) QLoRA** — the base is stored in 4-bit NF4: 7e9 × 0.5 B = 3.5 GB plus ~0.1 GB of
quantization constants, with the same ~0.6 GB adapter and 1–2 GB of activations: **≈5.5–7 GB** —
a 12 GB card, comfortably a 16 GB T4.

Rule of thumb: full FT ≈ 16 B/param, LoRA ≈ 2 B/param frozen + a negligible adapter, QLoRA ≈
0.5 B/param. The measured 0.5B LoRA peak of 9.12 GB is mostly activations (weights ≈ 1 GB),
which is the same reason batch size, not parameter count, drives memory in (b) and (c).
