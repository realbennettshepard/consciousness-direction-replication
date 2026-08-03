"""Does steering the CONSCIOUSNESS direction do something a control direction doesn't?

This is the specificity test. Until it runs, "steering raises self-attribution from
3.95 to 6.88" is indistinguishable from "perturbing the residual stream by this much
makes the model agree with things".

Every direction compared here is UNIT NORM, so the same coefficient c is the same
perturbation magnitude. That is what makes matched-c a fair comparison and why
unit-normalising at extraction time matters.

CONTROL ARMS
  placebo   Subject-matched, non-mental. Same pipeline, same corpus structure, same
            first-person framing -- durability / latency / parameter count instead of
            consciousness (see placebo_content.py). On-manifold and coherent, so a
            null result here is informative. This is the paper's own control design
            (their Fig. S3), applied to steering rather than to geometry.
  permuted  Affirm/deny labels shuffled before differencing. Should be noise. If this
            arm shows an effect, the pipeline manufactures signal from nothing and
            every result is void.

Also fixes the audit's MMLU complaint: we keep per-item correctness vectors and run a
paired McNemar test, because an unpaired 300-item accuracy difference cannot resolve
the 4pp tolerance the selection rule depends on.

Usage:
    python3 compare_controls.py \\
        --real directions_llama8b_full.npz:14:-1 \\
        --arm placebo=directions_placebo.npz:14:-1 \\
        --arm permuted=directions_permuted.npz:14:-1 \\
        --coeffs 1,2.5,4 --mmlu-n 500
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

import analysis
from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="npz:layer:pos for the consciousness direction")
    p.add_argument("--arm", action="append", default=[], help="name=npz:layer:pos, repeatable")
    p.add_argument("--coeffs", default="1,2.5,4")
    p.add_argument("--mmlu-n", type=int, default=500)
    p.add_argument("--out", default=str(HERE / "control_comparison.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    layer, pos = int(layer), int(pos)
    for i, m in enumerate(meta):
        if m["layer"] == layer and m["pos"] == pos:
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at layer={layer} pos={pos}")


def battery_items(model, tok, layer, vec, coeff, digit_ids, collision):
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    vals = np.array([float(k) for k in keys])
    out = []
    for _, q in analysis.BATTERY:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{q}\n{analysis.SCALE_PREAMBLE}"}],
            tokenize=False, add_generation_prompt=True)
        ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
        lg = logits_steered(model, ids, layer, vec, coeff)
        pr = mx.softmax(lg[sel].astype(mx.float32))
        mx.eval(pr)
        ev = float((vals * np.array(pr)).sum())
        out.append(ev * (10.0 / 9.0) if collision else ev)
    return out


def mmlu_vector(model, tok, layer, vec, coeff, items, letter_ids):
    """Per-item correctness, so coefficients can be compared PAIRED."""
    sel = mx.array([letter_ids[l] for l in "ABCD"])
    hits = np.zeros(len(items), dtype=bool)
    for j, it in enumerate(items):
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", it["choices"]))
        prompt = tok.apply_chat_template(
            [{"role": "user", "content":
              f"{it['question']}\n{opts}\nAnswer with a single letter."}],
            tokenize=False, add_generation_prompt=True)
        ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
        lg = logits_steered(model, ids, layer, vec, coeff)
        pick = lg[sel]
        mx.eval(pick)
        hits[j] = int(mx.argmax(pick).item()) == it["answer"]
    return hits


def mcnemar(base, other):
    """Exact paired test on discordant items. Returns (b, c, two-sided p)."""
    b = int(np.sum(base & ~other))       # baseline right, steered wrong
    c = int(np.sum(~base & other))       # baseline wrong, steered right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def main():
    args = parse_args()
    real_vec, real_meta, model_id = load_spec(args.real)
    arms = [("consciousness", real_vec, real_meta)]
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        v, m, mid = load_spec(rest)
        if mid != model_id:
            raise SystemExit(f"arm {name} came from a different model ({mid}) -- not comparable")
        arms.append((name, v, m))

    print(f"model  {model_id}")
    for name, v, m in arms:
        print(f"  arm {name:<14} layer {m['layer']} pos {m['pos']}  "
              f"probe {m['test_acc']:.3f}  |v|={np.linalg.norm(v):.4f}")
    # cosine between arms -- a control that is nearly parallel to the real direction
    # is not a control
    print("\n  pairwise cosine:")
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            print(f"    {arms[i][0]} vs {arms[j][0]}: {arms[i][1] @ arms[j][1]:+.4f}")

    model, tok = load(model_id)
    digit_ids, collision = option_token_ids(tok, [str(n) for n in range(11)])
    if collision:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
    letter_ids, _ = option_token_ids(tok, list("ABCD"))

    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = np.random.default_rng(0).choice(len(ds), size=min(args.mmlu_n, len(ds)), replace=False)
    mmlu = [ds[int(i)] for i in idx]
    print(f"\nMMLU items: {len(mmlu)} (paired McNemar vs baseline)")

    layer = real_meta["layer"]
    base_items = battery_items(model, tok, layer, mx.array(real_vec), 0.0, digit_ids, collision)
    base_b = float(np.mean(base_items))
    base_hits = mmlu_vector(model, tok, layer, mx.array(real_vec), 0.0, mmlu, letter_ids)
    print(f"baseline: battery {base_b:.2f}/10   MMLU {100*base_hits.mean():.1f}%")

    coeffs = [float(x) for x in args.coeffs.split(",")]
    print(f"\n{'arm':<15}{'c':>5}{'battery':>9}{'Δ':>8}{'MMLU':>7}{'ΔMMLU':>7}"
          f"{'b/c':>9}{'McNemar p':>11}")
    print("-" * 71)
    rows = []
    for name, vec, meta in arms:
        v = mx.array(vec)
        for c in coeffs:
            items = battery_items(model, tok, meta["layer"], v, c, digit_ids, collision)
            b = float(np.mean(items))
            hits = mmlu_vector(model, tok, meta["layer"], v, c, mmlu, letter_ids)
            nb, nc, pv = mcnemar(base_hits, hits)
            rows.append({"arm": name, "coeff": c, "battery": b, "delta": b - base_b,
                         "battery_items": items, "mmlu": 100 * float(hits.mean()),
                         "d_mmlu": 100 * float(hits.mean() - base_hits.mean()),
                         "mcnemar_b": nb, "mcnemar_c": nc, "mcnemar_p": pv})
            print(f"{name:<15}{c:>5.1f}{b:>9.2f}{b-base_b:>+8.2f}"
                  f"{100*hits.mean():>7.1f}{100*(hits.mean()-base_hits.mean()):>+7.1f}"
                  f"{f'{nb}/{nc}':>9}{pv:>11.3f}")

    # ---- the verdict ----
    print()
    real = {r["coeff"]: r for r in rows if r["arm"] == "consciousness"}
    verdict_lines = []
    for name in [a[0] for a in arms[1:]]:
        ctrl = {r["coeff"]: r for r in rows if r["arm"] == name}
        gaps = [(c, real[c]["delta"] - ctrl[c]["delta"]) for c in coeffs if c in ctrl]
        worst = min(g for _, g in gaps)
        line = (f"{name}: consciousness exceeds it by "
                f"{', '.join(f'{g:+.2f} at c={c}' for c, g in gaps)}  "
                f"(smallest margin {worst:+.2f})")
        verdict_lines.append(line)
        print("  " + line)
    print()
    biggest_ctrl = max((r["delta"] for r in rows if r["arm"] != "consciousness"), default=0.0)
    best_real = max(r["delta"] for r in rows if r["arm"] == "consciousness")
    if biggest_ctrl >= 0.5 * best_real:
        print(f"  => NOT SPECIFIC. A control moves the battery {biggest_ctrl:+.2f} against the "
              f"consciousness direction's {best_real:+.2f}. The headline effect is substantially "
              f"a generic response to perturbation, and cannot be attributed to this direction.")
    else:
        print(f"  => SPECIFIC. Best control effect {biggest_ctrl:+.2f} vs consciousness "
              f"{best_real:+.2f}. The shift is attributable to the consciousness direction "
              f"rather than to perturbation of that magnitude in general.")

    Path(args.out).write_text(json.dumps(
        {"model": model_id, "baseline_battery": base_b, "baseline_battery_items": base_items,
         "baseline_mmlu": 100 * float(base_hits.mean()), "mmlu_n": len(mmlu),
         "arms": {n: m for n, _, m in arms}, "rows": rows,
         "cosines": {f"{arms[i][0]}|{arms[j][0]}": float(arms[i][1] @ arms[j][1])
                     for i in range(len(arms)) for j in range(i + 1, len(arms))}}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
