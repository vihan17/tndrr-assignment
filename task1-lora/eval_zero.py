"""Fairer baseline: same base model, but the full 77-label list is injected into the prompt.

Shows whether the base model fails because of formatting or because it lacks the label
vocabulary. Reported in the write-up as a sanity check, not as the main comparison.

Usage:
    python eval_zero.py --n 300 --out results/zero
"""
import argparse, json, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import train
from train import BASE, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset("PolyAI/banking77")
    labels = ds["train"].features["label"].names
    # Patch the module-level system prompt so evaluate() uses the label list.
    train.SYS = ("You are a bank support ticket router. Pick exactly one category from this list:\n"
                 + ", ".join(labels)
                 + '\nReply with JSON only: {"category": "<one category from the list>"}')

    n = min(a.n, len(ds["test"]))
    test = ds["test"].select(range(n))
    texts = [x["text"] for x in test]
    golds = [labels[x["label"]] for x in test]

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")

    m = evaluate(model, tok, texts, golds, max_new=24, bs=24)
    s = {k: m[k] for k in ("n", "valid_json", "category_accuracy", "strict_json_exact")}
    print("ZERO-SHOT-WITH-LABELS:", json.dumps(s), flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    with open(os.path.join(a.out, "predictions.jsonl"), "w") as f:
        for t, g, raw, par in zip(texts, golds, m["raw"], m["parsed"]):
            f.write(json.dumps({"text": t, "gold": g, "pred_raw": raw.strip(),
                                "pred": par, "correct": par == g}) + "\n")


if __name__ == "__main__":
    main()
