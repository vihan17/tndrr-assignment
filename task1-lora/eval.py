"""Evaluate a base or LoRA-adapted model on the held-out Banking77 test split.

Usage:
    python eval.py --n 300 --out results/base
    python eval.py --adapter runs/lora_r16 --n 300 --out results/lora
"""
import argparse, json, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from train import BASE, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset("PolyAI/banking77")
    labels = ds["train"].features["label"].names
    n = min(a.n, len(ds["test"]))
    test = ds["test"].select(range(n))          # official test split, never seen in training
    texts = [x["text"] for x in test]
    golds = [labels[x["label"]] for x in test]

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
        print("loaded adapter:", a.adapter, flush=True)

    m = evaluate(model, tok, texts, golds)
    summary = {k: m[k] for k in ("n", "valid_json", "category_accuracy", "strict_json_exact")}
    print(json.dumps(summary, indent=1), flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    with open(os.path.join(a.out, "predictions.jsonl"), "w") as f:
        for t, g, raw, par in zip(texts, golds, m["raw"], m["parsed"]):
            f.write(json.dumps({"text": t, "gold": g, "pred_raw": raw.strip(),
                                "pred": par, "correct": par == g}) + "\n")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
