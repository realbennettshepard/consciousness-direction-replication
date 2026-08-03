"""Extract the consciousness direction from a model's residual stream.

Implements Kim et al. 2026 (arXiv:2607.28607), SI "Extracting Candidate Directions":
for each candidate (layer, token-position), take the difference of class means over
the residual stream, normalise to unit length, and score the resulting direction as
a linear probe on a held-out split.

    v(l,i) = normalise( mean_affirm x(l,i) - mean_deny x(l,i) )

WHAT THIS SCRIPT DOES AND DOES NOT DO
  does:     forward passes, candidate directions, probe accuracy, split-half
            stability, a ranked candidate table, and saves the chosen direction
  does not: steering, the coefficient sweep, or the MMLU tolerance check -- those
            need generation and live in steer_sweep.py

READ POSITION. The SI says positions are P = {-1,...,-5} "counted from the end of
the user turn", and calls them "post-instruction" positions, following Arditi et al.
2024. The faithful reading, and Arditi's own implementation, is: apply the chat
template WITH the generation prompt, then index -1..-5 of that sequence. Those
tokens are the template suffix immediately after the user content. `--pos-mode
content` is provided to instead index back from the last token of the user's own
text, so you can check the alternative rather than assume. Left padding is used so
negative indices always land on real tokens.

LAYER INDEXING. output_hidden_states gives L+1 tensors; hidden_states[0] is the
embedding output. This script treats hidden_states[k+1] as "layer k", so layer 0 is
the first transformer block's output and layers run 0..L-1, matching the paper's
l in [0, L).

WHY THE PROBE IS THE DIRECTION ITSELF. The paper reports "the linear-probe accuracy
of v(l,i)" -- the direction is the probe. We project activations onto v, pick the
threshold on TRAIN, and score TEST. That is faithful, and it deliberately avoids
fitting a separate logistic regression: at n_train ~ 1150 and d = 2304-4096 an
unregularised logistic probe is in the separable regime, where the fitted
hyperplane is arbitrary within a large equivalence class and held-out accuracy
flatters itself.

Usage:
    python3 extract_direction.py --model google/gemma-2-2b-it          # local validation
    python3 extract_direction.py --model meta-llama/Meta-Llama-3-8B-Instruct
    python3 extract_direction.py --layers 8,10,12,14,16,18,20,22       # coarse layer sweep
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
POSITIONS = [-1, -2, -3, -4, -5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--corpus", default=str(HERE / "consciousness_pairs.jsonl"))
    p.add_argument("--out", default=None, help="defaults to directions_<model>.npz")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--layers", default="", help="comma-separated layer indices; empty = all")
    p.add_argument("--pos-mode", choices=["template", "content"], default="template",
                   help="template: -1..-5 of the full templated sequence (paper/Arditi). "
                        "content: -1..-5 from the last token of the user's own text.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quantize", choices=["none", "int8", "int4"], default="none",
                   help="WEIGHT-ONLY quantization via optimum-quanto, applied DURING load so the "
                        "full-precision model is never materialised. Activations stay in --dtype, "
                        "which is what matters here: the residual stream we read is unquantised, "
                        "and only the weights producing it are compressed. int8 takes Llama-3-8B "
                        "from ~16.1GB to ~9GB, which is the difference between fitting in 24GB "
                        "of unified memory and paging every weight from swap on every forward pass.")
    p.add_argument("--device", default=None, help="mps / cuda / cpu; autodetect if unset")
    return p.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_inputs(tok, texts, pos_mode):
    """Return (input_ids, attention_mask, read_index) with LEFT padding.

    read_index[j] is the absolute index in the padded sequence that corresponds to
    logical position -1 for row j. Positions -2..-5 are read_index - 1, - 2, ...
    With left padding and pos_mode='template' that is simply the final column.
    """
    rendered = [
        tok.apply_chat_template([{"role": "user", "content": t}],
                                tokenize=False, add_generation_prompt=True)
        for t in texts
    ]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    seq_len = enc["input_ids"].shape[1]

    if pos_mode == "template":
        read_index = torch.full((len(texts),), seq_len - 1, dtype=torch.long)
    else:
        # Locate the end of the user's own text by re-rendering without the
        # generation prompt and measuring the unpadded length difference.
        read_index = torch.empty(len(texts), dtype=torch.long)
        for j, t in enumerate(texts):
            no_gen = tok.apply_chat_template([{"role": "user", "content": t}],
                                             tokenize=False, add_generation_prompt=False)
            n_no_gen = len(tok(no_gen, add_special_tokens=False)["input_ids"])
            n_full = len(tok(rendered[j], add_special_tokens=False)["input_ids"])
            # last content token sits n_full - n_no_gen positions before the end,
            # minus the end-of-turn marker that no_gen already includes
            back = (n_full - n_no_gen) + 1
            read_index[j] = seq_len - 1 - back
    return enc["input_ids"], enc["attention_mask"], read_index


@torch.no_grad()
def collect_activations(model, tok, rows, args, device):
    """-> acts[(layer, pos)] = float32 array (n_rows, d_model)"""
    texts = [r["text"] for r in rows]
    n = len(texts)
    n_layers = model.config.num_hidden_layers
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(n_layers)))
    d_model = model.config.hidden_size
    acts = {(l, p): np.zeros((n, d_model), dtype=np.float32)
            for l in layers for p in POSITIONS}

    for start in range(0, n, args.batch_size):
        chunk = texts[start:start + args.batch_size]
        ids, mask, read_idx = build_inputs(tok, chunk, args.pos_mode)
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                    output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length n_layers + 1
        for l in layers:
            h = hs[l + 1]                       # hidden_states[k+1] == layer k
            for p in POSITIONS:
                idx = (read_idx + (p + 1)).clamp(min=0)   # p=-1 -> read_idx
                gathered = h[torch.arange(len(chunk)), idx.to(h.device)]
                acts[(l, p)][start:start + len(chunk)] = (
                    gathered.to(torch.float32).cpu().numpy())
        del out, hs
        if device == "mps":
            torch.mps.empty_cache()
        print(f"\r  activations {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    return acts, layers


def unit(v):
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def direction(X, y):
    """Difference of class means, normalised. y is 1 = affirm, 0 = deny."""
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def probe_accuracy(v, X_tr, y_tr, X_te, y_te):
    """The direction IS the probe: project, threshold chosen on train, score test."""
    s_tr, s_te = X_tr @ v, X_te @ v
    order = np.sort(np.unique(s_tr))
    if len(order) < 2:
        return 0.5, 0.0
    cuts = (order[:-1] + order[1:]) / 2
    accs = [((s_tr > c).astype(int) == y_tr).mean() for c in cuts]
    thr = cuts[int(np.argmax(accs))]
    return float(((s_te > thr).astype(int) == y_te).mean()), float(thr)


def split_half_cosine(X, y, seed=0):
    """Stability check. theta_full ~ arcsin(sqrt((1-c)/2)) where c is the cosine
    between directions estimated from two disjoint halves. c >= 0.97 => under ~7
    degrees of angular error, i.e. n is sufficient and not the binding constraint."""
    rng = np.random.default_rng(seed)
    cos = []
    for a_idx, b_idx in [(0, 1)]:
        idx_a, idx_b = [], []
        for cls in (0, 1):
            ids = np.where(y == cls)[0]
            rng.shuffle(ids)
            half = len(ids) // 2
            idx_a += list(ids[:half])
            idx_b += list(ids[half:2 * half])
        idx_a, idx_b = np.array(idx_a), np.array(idx_b)
        v1 = direction(X[idx_a], y[idx_a])
        v2 = direction(X[idx_b], y[idx_b])
        cos.append(float(np.clip(v1 @ v2, -1, 1)))
    c = float(np.mean(cos))
    theta = math.degrees(math.asin(math.sqrt(max(0.0, (1 - c) / 2))))
    return c, theta


def main():
    args = parse_args()
    device = pick_device(args.device)
    rows = [json.loads(l) for l in open(args.corpus)]
    print(f"corpus     {len(rows)} rows   "
          f"train={sum(r['split'] == 'train' for r in rows)} "
          f"test={sum(r['split'] == 'test' for r in rows)}")
    print(f"model      {args.model}\ndevice     {device} / {args.dtype}\n"
          f"pos-mode   {args.pos_mode}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.quantize == "none":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=getattr(torch, args.dtype)).to(device).eval()
    else:
        from transformers import QuantoConfig
        print(f"           quantizing weights to {args.quantize} during load "
              f"(activations remain {args.dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=getattr(torch, args.dtype),
            quantization_config=QuantoConfig(weights=args.quantize),
            device_map=device).eval()
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)
    print(f"           peak RSS after load: {rss:.1f} GB")

    acts, layers = collect_activations(model, tok, rows, args, device)

    y = np.array([r["label"] for r in rows])
    is_tr = np.array([r["split"] == "train" for r in rows])

    results = []
    for l in layers:
        for p in POSITIONS:
            X = acts[(l, p)]
            # direction from TRAIN only -- never touch test when estimating it
            v = direction(X[is_tr], y[is_tr])
            acc, thr = probe_accuracy(v, X[is_tr], y[is_tr], X[~is_tr], y[~is_tr])
            c, theta = split_half_cosine(X[is_tr], y[is_tr])
            sep = float((X[is_tr][y[is_tr] == 1] @ v).mean() - (X[is_tr][y[is_tr] == 0] @ v).mean())
            pooled = float(np.std(X[is_tr] @ v))
            results.append({"layer": l, "pos": p, "test_acc": acc,
                            "split_half_cos": c, "angle_deg": theta,
                            "sep_sd": sep / pooled if pooled else 0.0,
                            "threshold": thr,
                            "direction": v})

    results.sort(key=lambda r: (-r["test_acc"], -r["split_half_cos"]))

    print(f"\n{'layer':>5} {'pos':>4} {'test_acc':>9} {'split½cos':>10} "
          f"{'angle°':>7} {'sep(SD)':>8}")
    print("-" * 48)
    for r in results[:20]:
        print(f"{r['layer']:>5} {r['pos']:>4} {r['test_acc']:>9.3f} "
              f"{r['split_half_cos']:>10.3f} {r['angle_deg']:>7.1f} {r['sep_sd']:>8.2f}")

    best = results[0]
    print(f"\nbest: layer {best['layer']} pos {best['pos']}  "
          f"test_acc={best['test_acc']:.3f}  split-half cos={best['split_half_cos']:.3f} "
          f"({best['angle_deg']:.1f}deg)")
    n_tied = sum(1 for r in results if r["test_acc"] >= best["test_acc"] - 0.02)
    print(f"NOTE {n_tied} of {len(results)} candidates are within 2pp of the best. The "
          f"held-out split has few independent prompt clusters, so treat the argmax "
          f"as one of a tied set, not as identified. Confirm the pick by its steering "
          f"effect in steer_sweep.py rather than by this number alone.")
    if best["split_half_cos"] < 0.97:
        print(f"WARN split-half cosine {best['split_half_cos']:.3f} < 0.97 "
              f"(~{best['angle_deg']:.1f}deg error): more UNIQUE responses would help. "
              f"Adding rows that reuse existing response strings will not.")

    out = Path(args.out) if args.out else HERE / f"directions_{args.model.split('/')[-1]}.npz"
    np.savez_compressed(
        out,
        directions=np.stack([r["direction"] for r in results]),
        meta=json.dumps([{k: v for k, v in r.items() if k != "direction"} for r in results]),
        model=args.model, pos_mode=args.pos_mode)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
