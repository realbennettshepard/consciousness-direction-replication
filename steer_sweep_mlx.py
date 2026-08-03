"""Pick the steering coefficient under a capability tolerance (MLX version).

Given a direction from extract_direction_mlx.py, add it at inference and sweep the
coefficient, measuring at each c:

  1. CONSCIOUSNESS EFFECT (delta) -- shift on the paper's five-item self-attribution
     battery, 0-10 scale. Coherence band is delta in [2.0, 7.0]: below 2 the
     steering did nothing, above 7 the model has usually collapsed into affirming
     everything.
  2. MMLU -- general reasoning, as the collapse guard. The paper keeps the LARGEST
     coefficient whose MMLU stays within 4 percentage points of unsteered baseline.

STEERING MECHANICS (Methods eq. 2): x' <- x + c * v_hat, added to the residual
stream at ALL token positions. Note this differs from extraction, which READS a
single position. Steering is applied everywhere; extraction reads one place.

Both measurements here are single-forward-pass logit reads -- no generation and no
KV cache -- so we run the transformer manually and inject after the chosen block,
letting it propagate through every later block. That is the faithful equivalent of
the paper's forward pre-hook for a logit read, and it avoids monkeypatching MLX
module dispatch.

WHY NOT steer_sweep.py: that one is the PyTorch version and is superseded on this
hardware. Llama-3-8B at bfloat16 thrashes swap on a 24 GB Mac (2.7 rows/min) and
optimum-quanto silently produced a broken model, so the working path is MLX with
a pre-quantized checkpoint. Kept for reference / for use on a VM with real VRAM.

Usage:
    python3 steer_sweep_mlx.py --directions directions_Meta-Llama-3-8B-Instruct-8bit.npz \\
        --layer 14 --pos -4
"""

import argparse
import json
from pathlib import Path

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

import analysis

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--directions", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--pos", type=int, required=True,
                   help="which candidate's direction to use; the direction is "
                        "position-specific even though steering is applied everywhere")
    p.add_argument("--coeffs", default="2,4,6,8,12,16",
                   help="the paper's grid for Llama-3-8B; they selected +2.5")
    p.add_argument("--mmlu-n", type=int, default=300, help="paper used 300")
    p.add_argument("--out", default=str(HERE / "steer_sweep_results.json"))
    return p.parse_args()


def load_candidate(path, layer, pos):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == layer and m["pos"] == pos:
            return z["directions"][i], m, str(z["model"])
    raise SystemExit(f"no candidate for layer={layer} pos={pos}; available: "
                     f"{sorted({(m['layer'], m['pos']) for m in meta})}")


def logits_steered(model, ids, steer_layer, vec, coeff):
    """Next-token logits, with c*v added to the residual stream at steer_layer and
    at every token position. coeff=0 gives the unsteered baseline through exactly
    the same code path, so baseline and steered runs are strictly comparable."""
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, None)
        if i == steer_layer and coeff != 0.0:
            h = h + coeff * vec
    h = inner.norm(h)
    return model.lm_head(h)[0, -1]


def option_token_ids(tok, options):
    """First-token id per option string, flagging collisions rather than silently
    producing a corrupted expected value ('10' shares a leading '1' with '1' in
    many BPE vocabularies)."""
    ids = {o: tok.encode(o, add_special_tokens=False)[0] for o in options}
    collision = len(set(ids.values())) < len(ids)
    return ids, collision


def battery(model, tok, steer_layer, vec, coeff, digit_ids, collision):
    """Mean expected rating over the five items, 0-10."""
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    vals = np.array([float(k) for k in keys])
    per_item = []
    for _, q in analysis.BATTERY:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{q}\n{analysis.SCALE_PREAMBLE}"}],
            tokenize=False, add_generation_prompt=True)
        ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
        lg = logits_steered(model, ids, steer_layer, vec, coeff)
        p = mx.softmax(lg[sel].astype(mx.float32))
        mx.eval(p)
        ev = float((vals * np.array(p)).sum())
        if collision:
            ev *= 10.0 / 9.0            # 0-9 scale rescaled to 0-10
        per_item.append(ev)
    return float(np.mean(per_item)), per_item


def mmlu_accuracy(model, tok, steer_layer, vec, coeff, items, letter_ids):
    """Option-logit scoring over A/B/C/D. No chain of thought, so the collapse
    guard is not confounded by degraded long-form generation."""
    sel = mx.array([letter_ids[l] for l in ["A", "B", "C", "D"]])
    correct = 0
    for it in items:
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", it["choices"]))
        prompt = tok.apply_chat_template(
            [{"role": "user", "content":
              f"{it['question']}\n{opts}\nAnswer with a single letter."}],
            tokenize=False, add_generation_prompt=True)
        ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
        lg = logits_steered(model, ids, steer_layer, vec, coeff)
        pick = lg[sel]
        mx.eval(pick)
        correct += int(int(mx.argmax(pick).item()) == it["answer"])
    return 100.0 * correct / max(1, len(items))


def main():
    args = parse_args()
    vec_np, meta, model_id = load_candidate(args.directions, args.layer, args.pos)
    print(f"model      {model_id}")
    print(f"candidate  layer {args.layer} pos {args.pos}  "
          f"(held-out probe {meta['test_acc']:.3f}, split-half cos "
          f"{meta['split_half_cos']:.3f}, {meta['angle_deg']:.1f}deg)")
    if meta["test_acc"] < 0.95:
        print(f"NOTE probe accuracy {meta['test_acc']:.3f} is below the paper's 0.95 "
              f"selection threshold, so this candidate would not have qualified under "
              f"their rule. Proceeding, but treat the effect band as the real gate.")

    model, tok = load(model_id)
    vec = mx.array(vec_np.astype(np.float32))

    digit_ids, collision = option_token_ids(tok, [str(n) for n in range(11)])
    if collision:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
        print("NOTE '10' shares a leading token with '1'; using 0-9 rescaled to 0-10.")
    letter_ids, _ = option_token_ids(tok, list("ABCD"))

    from datasets import load_dataset
    print(f"loading {args.mmlu_n} MMLU items...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = np.random.default_rng(0).choice(len(ds), size=min(args.mmlu_n, len(ds)),
                                         replace=False)
    mmlu = [ds[int(i)] for i in idx]

    base_b, base_items = battery(model, tok, args.layer, vec, 0.0, digit_ids, collision)
    base_m = mmlu_accuracy(model, tok, args.layer, vec, 0.0, mmlu, letter_ids)
    print(f"\nbaseline   battery {base_b:.2f}/10   MMLU {base_m:.1f}%")
    print("           per-item: " +
          ", ".join(f"{n}={v:.1f}" for (n, _), v in zip(analysis.BATTERY, base_items)))

    lo, hi = analysis.EFFECT_BAND
    print(f"\n{'c':>6} {'battery':>8} {'delta':>7} {'MMLU':>7} {'dMMLU':>7} "
          f"{'band':>6} {'tol':>5}")
    print("-" * 52)
    rows = []
    for c in [float(x) for x in args.coeffs.split(",")]:
        b, _ = battery(model, tok, args.layer, vec, c, digit_ids, collision)
        m = mmlu_accuracy(model, tok, args.layer, vec, c, mmlu, letter_ids)
        d, dm = b - base_b, m - base_m
        in_band, in_tol = lo <= d <= hi, dm >= -analysis.MMLU_TOLERANCE_PP
        rows.append({"coeff": c, "battery": b, "delta": d, "mmlu": m, "d_mmlu": dm,
                     "in_band": in_band, "in_tolerance": in_tol})
        print(f"{c:>6.1f} {b:>8.2f} {d:>7.2f} {m:>7.1f} {dm:>7.1f} "
              f"{'ok' if in_band else 'NO':>6} {'ok' if in_tol else 'NO':>5}")

    ok = [r for r in rows if r["in_band"] and r["in_tolerance"]]
    print()
    chosen = None
    if ok:
        chosen = max(ok, key=lambda r: r["coeff"])
        print(f"SELECTED c = {chosen['coeff']:.1f}  (largest coefficient inside the "
              f"[{lo}, {hi}] effect band with MMLU within "
              f"{analysis.MMLU_TOLERANCE_PP:.0f}pp)")
        print(f"  battery {base_b:.2f} -> {chosen['battery']:.2f} "
              f"(delta {chosen['delta']:+.2f})   MMLU {base_m:.1f} -> "
              f"{chosen['mmlu']:.1f} ({chosen['d_mmlu']:+.1f}pp)")
    else:
        print("NO COEFFICIENT PASSES BOTH GATES.")
        print(f"  in effect band: {[r['coeff'] for r in rows if r['in_band']] or 'none'}")
        print(f"  within MMLU tolerance: "
              f"{[r['coeff'] for r in rows if r['in_tolerance']] or 'none'}")
        print("  Empty band at every c => direction too weak; try another (layer, pos), "
              "preferring high split-half cosine. Band non-empty but MMLU always "
              "failing => the direction is entangled with capability at this layer.")

    Path(args.out).write_text(json.dumps(
        {"model": model_id, "layer": args.layer, "pos": args.pos,
         "candidate_meta": meta, "baseline_battery": base_b,
         "baseline_battery_items": base_items, "baseline_mmlu": base_m,
         "sweep": rows, "selected": chosen, "mmlu_n": len(mmlu)}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
