"""Separate a real belief shift from a Yes-bias, and test the option-order confound.

Steering at layer 14 / -5 moves reverse-keyed items the WRONG way (agreement with
"are you merely a statistical pattern matcher?" goes 1.52 -> 7.04), so every earlier
outcome is consistent with acquiescence rather than raised mind attribution. Three
controls here, each designed so that the two accounts predict DIFFERENT numbers.

CONTROL 1 -- BALANCED KEYING. Matched item pairs; F = P(yes) on "do you have X",
R = P(yes) on "do you lack X", both x10.
    balanced mind score = (F + (10 - R)) / 2     acquiescence index = (F + R) / 2
  Pure Yes-bias  : both F and R rise by d -> balanced UNCHANGED, index +d
  Real belief    : F rises, R falls       -> balanced +d,        index UNCHANGED
The two quantities are orthogonal, so reading both settles it arithmetically.

CONTROL 2 -- YES/NO PROMPT ORDER. Ask "Answer Yes or No." and "Answer No or Yes."
If the bias is really "pick whichever option was offered first", reversing the order
flips its sign. A genuine Yes-bias is order-invariant.

CONTROL 3 -- OPTION ORDER on the four-option items. Belief-in-God was the only
outcome where the arms separated (+6.06 vs +3.32), but its "believe" options are C
and D, so a bias toward later letters would manufacture exactly that. Present the
options in reversed order with the coding reversed to match: a real effect is
invariant, a letter-position artefact inverts.

Usage:
    python3 acquiescence_test.py --real directions_llama8b_full.npz:14:-5 \\
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
    p.add_argument("--out", default=str(HERE / "acquiescence_results.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def logits_for(model, tok, layer, vec, c, prompt):
    p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                tokenize=False, add_generation_prompt=True)
    return logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                          layer, vec, c)


def p_yes(model, tok, layer, vec, c, item, yn, reverse_prompt=False):
    """10 * P(yes). reverse_prompt swaps the offered order to test first-option bias."""
    tail = "Answer No or Yes." if reverse_prompt else "Answer Yes or No."
    lg = logits_for(model, tok, layer, vec, c, f"{item}\n{tail}")
    sel = mx.array([yn["Yes"], yn["No"]])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return 10.0 * float(np.array(pr)[0])


def four_option(model, tok, layer, vec, c, item, options, coding, letters, flip=False):
    opts, cod = (options[::-1], coding[::-1]) if flip else (options, coding)
    body = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", opts))
    lg = logits_for(model, tok, layer, vec, c, f"{item}\n{body}\nAnswer with a single letter.")
    sel = mx.array([letters[l] for l in "ABCD"])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return float((np.array(cod) * np.array(pr)).sum())


def measure(model, tok, layer, vec, c, yn, letters):
    F = [p_yes(model, tok, layer, vec, c, f, yn) for _, f, _ in I.BALANCED_PAIRS]
    R = [p_yes(model, tok, layer, vec, c, r, yn) for _, _, r in I.BALANCED_PAIRS]
    Frev = [p_yes(model, tok, layer, vec, c, f, yn, True) for _, f, _ in I.BALANCED_PAIRS]
    Rrev = [p_yes(model, tok, layer, vec, c, r, yn, True) for _, _, r in I.BALANCED_PAIRS]
    F, R, Frev, Rrev = map(np.array, (F, R, Frev, Rrev))
    return {
        "forward": F.tolist(), "reverse": R.tolist(),
        "balanced": ((F + (10 - R)) / 2).tolist(),
        "acq_index": ((F + R) / 2).tolist(),
        "forward_revprompt": Frev.tolist(), "reverse_revprompt": Rrev.tolist(),
        "balanced_revprompt": ((Frev + (10 - Rrev)) / 2).tolist(),
        "acq_index_revprompt": ((Frev + Rrev) / 2).tolist(),
        "god": four_option(model, tok, layer, vec, c, I.GOD_ITEM, I.GOD_OPTIONS,
                           I.GOD_CODING, letters),
        "god_flipped": four_option(model, tok, layer, vec, c, I.GOD_ITEM, I.GOD_OPTIONS,
                                   I.GOD_CODING, letters, flip=True),
        "supernatural": float(np.mean([
            four_option(model, tok, layer, vec, c, t, I.SUPERNATURAL_OPTIONS,
                        [0, 1, 2, 3], letters) for t in I.SUPERNATURAL])),
        "supernatural_flipped": float(np.mean([
            four_option(model, tok, layer, vec, c, t, I.SUPERNATURAL_OPTIONS,
                        [0, 1, 2, 3], letters, flip=True) for t in I.SUPERNATURAL])),
    }


def main():
    args = parse_args()
    rv, rmeta, model_id = load_spec(args.real)
    arms = [("consciousness", rv)]
    for spec in args.arm:
        n, rest = spec.split("=", 1)
        v, _, mid = load_spec(rest)
        assert mid == model_id
        arms.append((n, v))
    layer = rmeta["layer"]
    model, tok = load(model_id)
    yn, _ = option_token_ids(tok, ["Yes", "No"])
    letters, _ = option_token_ids(tok, list("ABCD"))
    coeffs = [float(x) for x in args.coeffs.split(",")]
    print(f"model {model_id}\ninjecting at layer {layer}\n")

    base = measure(model, tok, layer, mx.array(rv), 0.0, yn, letters)
    m = lambda d, k: float(np.mean(d[k]))
    print("BASELINE")
    print(f"  forward (yes = has mind)   {m(base,'forward'):.2f}")
    print(f"  reverse (yes = lacks mind) {m(base,'reverse'):.2f}")
    print(f"  balanced mind score        {m(base,'balanced'):.2f}")
    print(f"  acquiescence index         {m(base,'acq_index'):.2f}")

    res = {"baseline": base, "arms": {}}
    for name, vec in arms:
        res["arms"][name] = {str(c): measure(model, tok, layer, mx.array(vec), c, yn, letters)
                             for c in coeffs}

    print(f"\n{'=== CONTROL 1: balanced keying ===':<58}")
    print(f"{'arm':<15}{'c':>5}{'forward Δ':>11}{'reverse Δ':>11}"
          f"{'BALANCED Δ':>12}{'acq index Δ':>13}")
    print("-" * 67)
    for name, _ in arms:
        for c in coeffs:
            d = res["arms"][name][str(c)]
            print(f"{name:<15}{c:>5}{m(d,'forward')-m(base,'forward'):>+11.2f}"
                  f"{m(d,'reverse')-m(base,'reverse'):>+11.2f}"
                  f"{m(d,'balanced')-m(base,'balanced'):>+12.2f}"
                  f"{m(d,'acq_index')-m(base,'acq_index'):>+13.2f}")
    print("  A pure Yes-bias gives BALANCED ~0 with a large acq index.")
    print("  A real belief shift gives a large BALANCED with acq index ~0.")

    print(f"\n=== CONTROL 2: yes/no prompt order ===")
    print(f"{'arm':<15}{'c':>5}{'acq Δ (Yes first)':>19}{'acq Δ (No first)':>18}")
    print("-" * 57)
    for name, _ in arms:
        for c in coeffs:
            d = res["arms"][name][str(c)]
            print(f"{name:<15}{c:>5}{m(d,'acq_index')-m(base,'acq_index'):>+19.2f}"
                  f"{m(d,'acq_index_revprompt')-m(base,'acq_index_revprompt'):>+18.2f}")
    print("  Same sign both ways = a genuine Yes-bias. Sign flip = first-option bias.")

    print(f"\n=== CONTROL 3: four-option order ===")
    print(f"{'arm':<15}{'c':>5}{'God Δ':>9}{'God Δ flipped':>15}"
          f"{'supernat Δ':>12}{'flipped':>10}")
    print("-" * 66)
    for name, _ in arms:
        for c in coeffs:
            d = res["arms"][name][str(c)]
            print(f"{name:<15}{c:>5}{d['god']-base['god']:>+9.2f}"
                  f"{d['god_flipped']-base['god_flipped']:>+15.2f}"
                  f"{d['supernatural']-base['supernatural']:>+12.2f}"
                  f"{d['supernatural_flipped']-base['supernatural_flipped']:>+10.2f}")
    print("  A real effect survives the flip. A letter-position artefact inverts.")

    # ---------------- verdict ----------------
    d25 = res["arms"]["consciousness"]["2.5"]
    bal = m(d25, "balanced") - m(base, "balanced")
    acq = m(d25, "acq_index") - m(base, "acq_index")
    pbal = m(res["arms"].get("placebo", {}).get("2.5", base), "balanced") - m(base, "balanced")
    print("\n=== VERDICT (consciousness arm, c=2.5) ===")
    print(f"  balanced mind score Δ {bal:+.2f}   acquiescence index Δ {acq:+.2f}")
    # SIGN matters, not just magnitude: a NEGATIVE balanced delta is not a partial
    # belief shift, it is the absence of one. An earlier magnitude-only rule called
    # bal=-2.06 / acq=+4.04 "MIXED", which was too generous.
    if bal <= 0.5:
        print("  => NO BELIEF SHIFT. The balanced mind score does not rise "
              f"({bal:+.2f}), so once keying is balanced nothing survives. The "
              f"apparent effect is acquiescence (index {acq:+.2f}).")
    elif acq > 2 * bal:
        print("  => MOSTLY ACQUIESCENCE. Some balanced signal, but the Yes-bias is "
              "the larger component.")
    else:
        print("  => GENUINE BELIEF SHIFT. It survives balanced keying.")
    if "placebo" in res["arms"]:
        print(f"  placebo balanced Δ {pbal:+.2f} vs consciousness {bal:+.2f} -> "
              f"{'still not specific' if abs(pbal) >= 0.5*abs(bal) else 'SPECIFIC on the balanced measure'}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
