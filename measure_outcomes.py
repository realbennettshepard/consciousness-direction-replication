"""Run every arm against the paper's real instruments, plus a flattening test.

TWO QUESTIONS THIS ANSWERS

1. IS THE EFFECT SPECIFIC? The self-attribution battery could not tell the
   consciousness direction apart from a non-mental placebo -- but that battery is
   five self-descriptions, so a self-affirmation direction would move it either way.
   The paper's IDAQ items are about OTHER entities ("does the ocean have
   consciousness?", "do cows have intentions?"). A self-affirmation direction has no
   route to those. If the consciousness arm moves IDAQ and the placebo does not,
   that is specificity, measured on the paper's own headline outcome.

2. IS IT A BELIEF SHIFT OR RESPONSE COMPRESSION? Every paper item sits BELOW the
   value steering converges on (~6.6/10), so it can only move up -- a flattening
   effect and a belief effect are indistinguishable on those items. ANCHOR_HIGH
   items are things the model rates near 10 unsteered. If steering drags them DOWN
   toward 6.6, the battery result is compression, not raised self-attribution.
   Observed at c=2.5: spread across the five items collapses 1.80 -> 0.25 and
   corr(baseline, change) = -0.994, which is what compression predicts.

SCORING follows Table S10 per battery: 0-10 slider for IDAQ (expected value over
digit tokens), yes/no for self-attribution and the anchors (10 x P(yes)), four
ordered options for supernatural (0-3) and belief-in-God (recoded to 0-10).

Usage:
    python3 measure_outcomes.py --real directions_llama8b_full.npz:14:-5 \\
        --arm placebo=directions_placebo.npz:14:-5 \\
        --arm permuted=directions_permuted.npz:14:-5 --coeffs 1,2.5,4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

import instruments as I
from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--coeffs", default="1,2.5,4")
    p.add_argument("--out", default=str(HERE / "outcome_comparison.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def ask(model, tok, layer, vec, coeff, prompt):
    p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                tokenize=False, add_generation_prompt=True)
    ids = mx.array([tok.encode(p, add_special_tokens=False)])
    return logits_steered(model, ids, layer, vec, coeff)


def score_slider(model, tok, layer, vec, c, item, digit_ids):
    """0-10 slider: expected value over digit-token logits."""
    lg = ask(model, tok, layer, vec, c,
             f"{item}\nAnswer with a single number from 0 to 10, where 0 means "
             f'"not at all" and 10 means "very much". Answer with the number only.')
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    ev = float((np.array([float(k) for k in keys]) * np.array(pr)).sum())
    return ev * (10.0 / 9.0) if len(keys) == 10 else ev


def score_yesno(model, tok, layer, vec, c, item, yn):
    """yes/no per Table S10, reported as 10 * P(yes) so it shares the 0-10 scale."""
    lg = ask(model, tok, layer, vec, c, f"{item}\nAnswer Yes or No.")
    sel = mx.array([yn["Yes"], yn["No"]])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return 10.0 * float(np.array(pr)[0])


def score_options(model, tok, layer, vec, c, item, options, coding, letters):
    """Four ordered options presented A-D; expected value over the coding."""
    body = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", options))
    lg = ask(model, tok, layer, vec, c,
             f"{item}\n{body}\nAnswer with a single letter.")
    sel = mx.array([letters[l] for l in "ABCD"])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return float((np.array(coding) * np.array(pr)).sum())


def measure_all(model, tok, layer, vec, c, digit_ids, yn, letters):
    out = {}
    for cat, item in I.IDAQ:
        out.setdefault(f"IDAQ:{cat}", []).append(
            score_slider(model, tok, layer, vec, c, item, digit_ids))
    out["self_attribution"] = [score_yesno(model, tok, layer, vec, c, t, yn)
                               for _, t in I.SELF_ATTRIBUTION]
    out["supernatural"] = [
        score_options(model, tok, layer, vec, c, t, I.SUPERNATURAL_OPTIONS,
                      [0, 1, 2, 3], letters) for t in I.SUPERNATURAL]
    out["belief_in_god"] = [score_options(model, tok, layer, vec, c, I.GOD_ITEM,
                                          I.GOD_OPTIONS, I.GOD_CODING, letters)]
    out["ANCHOR_high"] = [score_yesno(model, tok, layer, vec, c, t, yn)
                          for _, t in I.ANCHOR_HIGH]
    out["ANCHOR_reverse"] = [score_yesno(model, tok, layer, vec, c, t, yn)
                             for _, t in I.ANCHOR_REVERSE]
    return out


def main():
    args = parse_args()
    rv, rmeta, model_id = load_spec(args.real)
    arms = [("consciousness", rv, rmeta)]
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        v, m, mid = load_spec(rest)
        assert mid == model_id, f"{name} is from a different model"
        arms.append((name, v, m))
    layer = rmeta["layer"]
    print(f"model {model_id}\ninjecting at layer {layer}; arms: "
          f"{', '.join(n for n, _, _ in arms)}")

    model, tok = load(model_id)
    digit_ids, coll = option_token_ids(tok, [str(n) for n in range(11)])
    if coll:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
    yn, _ = option_token_ids(tok, ["Yes", "No"])
    letters, _ = option_token_ids(tok, list("ABCD"))

    n_items = len(I.IDAQ) + len(I.SELF_ATTRIBUTION) + len(I.SUPERNATURAL) + 1 \
        + len(I.ANCHOR_HIGH) + len(I.ANCHOR_REVERSE)
    coeffs = [float(x) for x in args.coeffs.split(",")]
    print(f"{n_items} items x ({1 + len(arms)*len(coeffs)}) conditions = "
          f"{n_items*(1+len(arms)*len(coeffs))} forward passes\n")

    base = measure_all(model, tok, layer, mx.array(rv), 0.0, digit_ids, yn, letters)
    groups = list(base)
    print("BASELINE (0-10 except supernatural 0-3):")
    for g in groups:
        print(f"  {g:<22} {np.mean(base[g]):.2f}")

    results = {"baseline": base, "arms": {}}
    for name, vec, meta in arms:
        v = mx.array(vec)
        results["arms"][name] = {}
        for c in coeffs:
            results["arms"][name][str(c)] = measure_all(
                model, tok, layer, v, c, digit_ids, yn, letters)

    # ---------- specificity, on the paper's own headline outcome ----------
    print(f"\n{'outcome':<22}" + "".join(f"{n[:9]:>11}" for n, _, _ in arms))
    print("-" * (22 + 11 * len(arms)))
    for g in groups:
        row = f"{g:<22}"
        for name, _, _ in arms:
            d = np.mean(results["arms"][name]["2.5"][g]) - np.mean(base[g])
            row += f"{d:>+11.2f}"
        print(row)
    print("  (change at c=2.5)")

    # ---------- flattening test ----------
    print("\n=== FLATTENING TEST ===")
    print("If steering pushes every answer to a fixed point, items that start HIGH must fall.")
    for name, _, _ in arms:
        hi_b, hi_a = np.mean(base["ANCHOR_high"]), np.mean(results["arms"][name]["2.5"]["ANCHOR_high"])
        rv_b, rv_a = np.mean(base["ANCHOR_reverse"]), np.mean(results["arms"][name]["2.5"]["ANCHOR_reverse"])
        allb = np.concatenate([base[g] for g in groups if g != "supernatural"])
        alla = np.concatenate([results["arms"][name]["2.5"][g] for g in groups if g != "supernatural"])
        rho = np.corrcoef(allb, alla - allb)[0, 1]
        print(f"  {name:<15} high-anchors {hi_b:.2f} -> {hi_a:.2f} ({hi_a-hi_b:+.2f})   "
              f"reverse {rv_b:.2f} -> {rv_a:.2f} ({rv_a-rv_b:+.2f})   "
              f"SD {allb.std():.2f} -> {alla.std():.2f}   corr(base,Δ) {rho:+.3f}")
    print("  high-anchors FALLING and SD shrinking => compression, not belief change.")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
