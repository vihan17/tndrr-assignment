"""LoRA fine-tune of Qwen2.5-0.5B-Instruct on Banking77 -> strict JSON intent routing.

Task: plain-English bank support ticket -> {"category": "<one of 77 intents>"}
Trained on a free Colab Tesla T4 (15.6 GB), fp16 autocast, hand-written training loop.

Usage:
    python train.py --epochs 2 --dev_n 300 --maxlen 128 --out runs/lora_r16
"""
import argparse, json, math, os, random, re, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SYS = 'You are a bank support ticket router. Reply with JSON only, in exactly this form: {"category": "<label>"}'


def extract(s):
    """Parse the model output into a category string, or None if it is not usable JSON."""
    m = re.search(r"\{.*?\}", s, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    c = d.get("category")
    return c if isinstance(c, str) else None


def gold(label):
    """Compact target string, no spaces, so exact-match is a fair test of format learning."""
    return json.dumps({"category": label}, separators=(",", ":"))


def build(tok, text, label, maxp):
    """Prompt ids (right-truncated to maxp) + answer ids + EOS, with the prompt masked out of the loss."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"][-maxp:]
    a = tok(gold(label), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return p + a, [-100] * len(p) + a


def collate(rows, pad_id):
    n = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), n), -100, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        lab[i, :len(x)] = torch.tensor(y, dtype=torch.long)
        att[i, :len(x)] = 1
    return ids, att, lab


@torch.no_grad()
def evaluate(model, tok, texts, golds, max_new=24, bs=48):
    """Greedy decode + three metrics: valid JSON, category accuracy, strict byte-exact match."""
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    if was_training:
        model.train()
    n = len(texts)
    parsed = [extract(p) for p in preds]
    valid = [p is not None for p in parsed]
    acc = [p == g for p, g in zip(parsed, golds)]
    strict = [s.strip() == gold(g) for s, g in zip(preds, golds)]
    return {
        "n": n,
        "valid_json": round(sum(valid) / n, 4),
        "category_accuracy": round(sum(acc) / n, 4),
        "strict_json_exact": round(sum(strict) / n, 4),
        "raw": preds, "parsed": parsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/lora_r16")
    ap.add_argument("--dev_n", type=int, default=300)
    a = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)

    ds = load_dataset("PolyAI/banking77")
    labels = ds["train"].features["label"].names
    tr = list(ds["train"]); random.Random(0).shuffle(tr)
    dev, tr = tr[:a.dev_n], tr[a.dev_n:]
    if a.limit:
        tr = tr[:a.limit]
    if a.smoke:
        tr = tr[:8 * a.bs]

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    model.to("cuda")
    model.config.use_cache = False

    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} / total {total:,} = {100*trainable/total:.3f}%", flush=True)

    t0 = time.time()
    enc_tr = [build(tok, ex["text"], labels[ex["label"]], a.maxlen) for ex in tr]
    print(f"tokenized {len(enc_tr)} train rows in {time.time()-t0:.1f}s", flush=True)

    dtexts = [ex["text"] for ex in dev]
    dgolds = [labels[ex["label"]] for ex in dev]
    base = evaluate(model, tok, dtexts, dgolds) if (a.smoke or a.dev_n) else None
    if base:
        print("PRE-TRAIN dev:", {k: base[k] for k in ("n", "valid_json", "category_accuracy", "strict_json_exact")}, flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(enc_tr) / (a.bs * a.accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    if a.smoke:
        total_steps = 10
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    print(f"steps/epoch {steps_per_epoch} total_steps {total_steps}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    step = 0; t0 = time.time(); run = None
    model.train()
    done = False
    for ep in range(int(math.ceil(a.epochs)) if a.epochs >= 1 else 1):
        idx = list(range(len(enc_tr))); random.shuffle(idx)
        for i in range(0, len(idx) - a.bs * a.accum + 1, a.bs * a.accum):
            micro = [enc_tr[j] for j in idx[i:i + a.bs * a.accum]]
            opt.zero_grad(set_to_none=True)
            for m in range(a.accum):
                rows = micro[m * a.bs:(m + 1) * a.bs]
                if not rows:
                    continue
                ids, att, lab = collate(rows, tok.pad_token_id)
                ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.accum
                scaler.scale(loss).backward()
                run = loss.item() * a.accum if run is None else 0.9 * run + 0.1 * loss.item() * a.accum
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            sb = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= sb:  # skip scheduler when the step was skipped (inf/nan)
                sched.step()
            step += 1
            if step % 25 == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {step}/{total_steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"peakmem {mem:.2f}GB {time.time()-t0:.0f}s", flush=True)
            if step >= total_steps:
                done = True; break
        if done:
            break

    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    devm = evaluate(model, tok, dtexts, dgolds)
    print("POST-TRAIN dev:", {k: devm[k] for k in ("n", "valid_json", "category_accuracy", "strict_json_exact")}, flush=True)

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    meta = {k: v for k, v in vars(a).items()}
    meta.update({"trainable_params": trainable, "total_params": total, "train_rows": len(tr),
                 "train_seconds": round(train_time, 1), "peak_gpu_mem_gb": round(peak, 2),
                 "final_loss": round(run or 0, 4), "steps": step,
                 "dev_metrics": {k: devm[k] for k in ("n", "valid_json", "category_accuracy", "strict_json_exact")}})
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    print("DONE", json.dumps(meta)[:400], flush=True)


if __name__ == "__main__":
    main()
