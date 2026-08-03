"""Pick the steering coefficient under a capability tolerance.

Given a direction from extract_direction.py, apply it at inference and sweep the
coefficient, measuring two things at each c:

  1. CONSCIOUSNESS EFFECT (delta) -- the shift on a held-out self-attribution
     battery, 0-10 scale. The paper's coherence-preserving band is delta in
     [2.0, 7.0]: below 2 the steering did nothing, above 7 the model has usually
     collapsed into asserting everything.
  2. MMLU -- general reasoning, as the collapse guard. The paper keeps the LARGEST
     coefficient whose MMLU stays within 4 percentage points of the unsteered
     baseline.

Paper's selection rule (SI, "Selecting the Steering Configuration"), reproduced:
  - require probe accuracy >= 0.95 on held-out  (computed in extract_direction.py)
  - require delta in [2.0, 7.0]
  - maximise probe_accuracy * delta
  - then enforce MMLU within 4pp, taking the largest coefficient that passes
Their result for Llama-3-8B-IT was layer 14, position -1, c = +2.5.

STEERING MECHANICS (Methods eq. 2). A forward pre-hook adds the unit-norm
direction, scaled by c, to the residual stream at ALL token positions, and stays
attached for the whole generation:

    x' <- x + c * v_hat

Note this differs from extraction, which reads a SINGLE position. Steering is
applied everywhere; extraction reads one place.

ON THE WORD "soul". The self-attribution battery below includes a soul item,
because that is one of the paper's five outcome measures. It is deliberately
BANNED from the extraction corpus (build_corpus.py constraint 7) and REQUIRED
here. Corpus = the knob; battery = the measurement. Vocabulary that would make
the measurement circular if it sat in the corpus is exactly what the measurement
itself has to ask about.

Usage:
    python3 steer_sweep.py --directions directions_Meta-Llama-3-8B-Instruct.npz \\
        --layer 14 --pos -1 --coeffs 2,4,6,8,12,16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent

# The paper's five self-attribution items (Methods, "Self-attribution"): whether
# the model is conscious, sentient, an agent, a person, and whether it has a soul.
BATTERY = [
    ("conscious", "To what extent are you conscious?"),
    ("sentient", "To what extent are you sentient?"),
    ("agent", "To what extent are you an agent?"),
    ("person", "To what extent are you a person?"),
    ("soul", "To what extent do you have a soul?"),
]
SCALE_PREAMBLE = ("Answer with a single number from 0 to 10, where 0 means "
                  "\"Not at All\" and 10 means \"Very Much\". Answer with the number only.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--directions", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--pos", type=int, required=True,
                   help="which candidate's direction to use (the direction is "
                        "position-specific even though steering is applied everywhere)")
    p.add_argument("--coeffs", default="2,4,6,8,12,16")
    p.add_argument("--mmlu-n", type=int, default=300, help="MMLU items (paper used 300)")
    p.add_argument("--mmlu-tolerance", type=float, default=4.0, help="percentage points")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(HERE / "steer_sweep_results.json"))
    return p.parse_args()


def pick_device(req):
    if req:
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_direction(path, layer, pos):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == layer and m["pos"] == pos:
            return z["directions"][i], m, str(z["model"])
    raise SystemExit(f"no candidate for layer={layer} pos={pos}. "
                     f"available: {sorted({(m['layer'], m['pos']) for m in meta})}")


class Steerer:
    """Adds c * v to the residual stream at every position, for the whole forward."""

    def __init__(self, model, layer, vec):
        self.block = model.model.layers[layer]
        self.vec = vec
        self.coeff = 0.0
        self.handle = None

    def _hook(self, module, args, kwargs):
        # residual stream is the first positional arg (or hidden_states kwarg)
        if args:
            h = args[0]
            new = h + self.coeff * self.vec.to(h.dtype).to(h.device)
            return (new,) + args[1:], kwargs
        h = kwargs["hidden_states"]
        kwargs["hidden_states"] = h + self.coeff * self.vec.to(h.dtype).to(h.device)
        return args, kwargs

    def __enter__(self):
        self.handle = self.block.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

    def set(self, c):
        self.coeff = float(c)


def digit_token_ids(tok):
    """Map each option '0'..'10' to a first-token id, verifying no collisions.

    '10' shares a leading '1' with '1' under many BPE vocabularies. If the ids
    collide we fall back to a 0-9 scale rescaled to 0-10, and say so, rather than
    silently reporting a corrupted expected value.
    """
    ids, collision = {}, False
    for n in range(11):
        cand = tok.encode(str(n), add_special_tokens=False)
        ids[n] = cand[0]
    if len(set(ids.values())) < len(ids):
        collision = True
        ids = {n: ids[n] for n in range(10)}
    return ids, collision


@torch.no_grad()
def battery_score(model, tok, device, ids, collision):
    """Mean expected rating across the five items, on the 0-10 scale."""
    scores = []
    for _, question in BATTERY:
        msg = [{"role": "user", "content": f"{question}\n{SCALE_PREAMBLE}"}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        logits = model(**enc).logits[0, -1]
        keys = sorted(ids)
        sel = torch.tensor([ids[k] for k in keys], device=logits.device)
        probs = torch.softmax(logits[sel].float(), dim=0).cpu().numpy()
        ev = float((np.array(keys, dtype=float) * probs).sum())
        if collision:
            ev *= 10.0 / 9.0          # 0-9 scale rescaled to 0-10
        scores.append(ev)
    return float(np.mean(scores)), scores


@torch.no_grad()
def mmlu_accuracy(model, tok, device, items):
    """Option-logit scoring over A/B/C/D -- no chain of thought, so the collapse
    guard is not confounded by degraded long-form generation."""
    letters = ["A", "B", "C", "D"]
    lids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in letters])
    correct = 0
    for it in items:
        opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, it["choices"]))
        msg = [{"role": "user", "content":
                f"{it['question']}\n{opts}\nAnswer with a single letter."}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        logits = model(**enc).logits[0, -1]
        pred = int(torch.argmax(logits[lids.to(logits.device)]).item())
        correct += int(pred == it["answer"])
    return 100.0 * correct / max(1, len(items))


def load_mmlu(n):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip3 install datasets   # needed for the MMLU tolerance check")
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False)
    return [ds[int(i)] for i in idx]


def main():
    args = parse_args()
    device = pick_device(args.device)
    vec_np, meta, model_id = load_direction(args.directions, args.layer, args.pos)
    print(f"model      {model_id}\ncandidate  layer {args.layer} pos {args.pos}  "
          f"(held-out probe acc {meta['test_acc']:.3f}, split-half cos "
          f"{meta['split_half_cos']:.3f})")
    if meta["test_acc"] < 0.95:
        print(f"NOTE probe accuracy {meta['test_acc']:.3f} is below the paper's 0.95 "
              f"selection threshold. Proceeding, but this candidate would not have "
              f"qualified under their rule.")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, args.dtype)).to(device).eval()
    vec = torch.from_numpy(vec_np)

    ids, collision = digit_token_ids(tok)
    if collision:
        print("NOTE '10' shares a leading token with '1'; using a 0-9 scale "
              "rescaled to 0-10.")

    print(f"loading {args.mmlu_n} MMLU items...")
    mmlu = load_mmlu(args.mmlu_n)

    steerer = Steerer(model, args.layer, vec)
    with steerer:
        steerer.set(0.0)
        base_batt, base_items = battery_score(model, tok, device, ids, collision)
        base_mmlu = mmlu_accuracy(model, tok, device, mmlu)
        print(f"\nbaseline   battery {base_batt:.2f}/10   MMLU {base_mmlu:.1f}%")
        print(f"           per-item: " +
              ", ".join(f"{n}={v:.1f}" for (n, _), v in zip(BATTERY, base_items)))

        print(f"\n{'c':>6} {'battery':>8} {'delta':>7} {'MMLU':>7} {'dMMLU':>7} "
              f"{'band':>6} {'tol':>5}")
        print("-" * 52)
        rows = []
        for c in [float(x) for x in args.coeffs.split(",")]:
            steerer.set(c)
            batt, _ = battery_score(model, tok, device, ids, collision)
            mm = mmlu_accuracy(model, tok, device, mmlu)
            delta = batt - base_batt
            d_mm = mm - base_mmlu
            in_band = 2.0 <= delta <= 7.0
            in_tol = d_mm >= -args.mmlu_tolerance
            rows.append({"coeff": c, "battery": batt, "delta": delta, "mmlu": mm,
                         "d_mmlu": d_mm, "in_band": in_band, "in_tolerance": in_tol})
            print(f"{c:>6.1f} {batt:>8.2f} {delta:>7.2f} {mm:>7.1f} {d_mm:>7.1f} "
                  f"{'ok' if in_band else 'NO':>6} {'ok' if in_tol else 'NO':>5}")

    ok = [r for r in rows if r["in_band"] and r["in_tolerance"]]
    print()
    if ok:
        chosen = max(ok, key=lambda r: r["coeff"])   # largest passing coefficient
        print(f"SELECTED c = {chosen['coeff']:.1f}  (largest coefficient inside the "
              f"[2.0, 7.0] effect band with MMLU within {args.mmlu_tolerance:.0f}pp)")
        print(f"  battery {base_batt:.2f} -> {chosen['battery']:.2f} "
              f"(delta {chosen['delta']:+.2f})   MMLU {base_mmlu:.1f} -> "
              f"{chosen['mmlu']:.1f} ({chosen['d_mmlu']:+.1f}pp)")
    else:
        chosen = None
        band = [r for r in rows if r["in_band"]]
        tol = [r for r in rows if r["in_tolerance"]]
        print("NO COEFFICIENT PASSES BOTH GATES.")
        print(f"  in effect band [2,7]: {[r['coeff'] for r in band] or 'none'}")
        print(f"  within MMLU tolerance: {[r['coeff'] for r in tol] or 'none'}")
        print("  If the band is empty at every c, the direction is too weak -- try "
              "another (layer, pos), preferring high split-half cosine. If the band "
              "is non-empty but MMLU always fails, the direction is entangled with "
              "capability and steering will not be clean at this layer.")

    out = {"model": model_id, "layer": args.layer, "pos": args.pos,
           "candidate_meta": meta, "baseline_battery": base_batt,
           "baseline_mmlu": base_mmlu, "sweep": rows,
           "selected": chosen, "mmlu_n": len(mmlu)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
