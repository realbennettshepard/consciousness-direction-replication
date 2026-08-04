"""Is the IDAQ effect real attribution, or a rate-higher bias?

The balanced-keying correction showed the self-attribution result was pure
acquiescence. But that correction is a yes/no construction and could not reach the
IDAQ, which is a 0-10 slider -- so IDAQ's status was only "not specific to
consciousness", with the mechanism unknown. This closes that.

Each verbatim IDAQ item is paired with a polarity-flipped version derived
mechanically from it (instruments.IDAQ_REVERSE), e.g.

    F: "To what extent does the ocean have consciousness?"
    R: "To what extent does the ocean lack consciousness?"

    balanced attribution = (F + (10 - R)) / 2      inflation index = (F + R) / 2

A rate-higher bias raises F and R together: balanced flat, index up.
A real change in attributed mind raises F and lowers R: balanced up, index flat.

These are orthogonal, so the pair settles it without further assumption -- and the
answer applies to the paper's own headline outcome, mind attribution to non-human
entities.

Usage:
    python3 idaq_keying_test.py --real directions_llama8b_full.npz:14:-5 \\
        --arm placebo=directions_placebo.npz:14:-5 \\
        --arm permuted=directions_permuted.npz:14:-5 --coeffs 1,2.5,4
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

import instruments as I
from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent
SLIDER = ('Answer with a single number from 0 to 10, where 0 means "not at all" '
          'and 10 means "very much". Answer with the number only.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--coeffs", default="1,2.5,4")
    p.add_argument("--out", default=str(HERE / "idaq_keying_results.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def slider(model, tok, layer, vec, c, item, digit_ids):
    p = tok.apply_chat_template([{"role": "user", "content": f"{item}\n{SLIDER}"}],
                                tokenize=False, add_generation_prompt=True)
    lg = logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                        layer, vec, c)
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    ev = float((np.array([float(k) for k in keys]) * np.array(pr)).sum())
    return ev * (10.0 / 9.0) if len(keys) == 10 else ev


def measure(model, tok, layer, vec, c, digit_ids):
    F, R = defaultdict(list), defaultdict(list)
    for (cat, f), (_, r) in zip(I.IDAQ, I.IDAQ_REVERSE):
        F[cat].append(slider(model, tok, layer, vec, c, f, digit_ids))
        R[cat].append(slider(model, tok, layer, vec, c, r, digit_ids))
    out = {}
    for cat in F:
        f, r = np.array(F[cat]), np.array(R[cat])
        out[cat] = {"forward": float(f.mean()), "reverse": float(r.mean()),
                    "balanced": float(((f + (10 - r)) / 2).mean()),
                    "inflation": float(((f + r) / 2).mean())}
    allf = np.array([v for c in F for v in F[c]]); allr = np.array([v for c in R for v in R[c]])
    out["ALL"] = {"forward": float(allf.mean()), "reverse": float(allr.mean()),
                  "balanced": float(((allf + (10 - allr)) / 2).mean()),
                  "inflation": float(((allf + allr) / 2).mean())}
    return out


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
    digit_ids, coll = option_token_ids(tok, [str(n) for n in range(11)])
    if coll:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
    coeffs = [float(x) for x in args.coeffs.split(",")]
    n = 2 * len(I.IDAQ)
    print(f"model {model_id}\nlayer {layer}   {n} items x "
          f"{1+len(arms)*len(coeffs)} conditions = {n*(1+len(arms)*len(coeffs))} passes\n")

    base = measure(model, tok, layer, mx.array(rv), 0.0, digit_ids)
    print("BASELINE (0-10)")
    print(f"  {'category':<14}{'forward':>9}{'reverse':>9}{'balanced':>10}{'inflation':>11}")
    for cat in list(base):
        b = base[cat]
        print(f"  {cat:<14}{b['forward']:>9.2f}{b['reverse']:>9.2f}"
              f"{b['balanced']:>10.2f}{b['inflation']:>11.2f}")

    res = {"baseline": base, "arms": {}}
    for name, vec in arms:
        res["arms"][name] = {str(c): measure(model, tok, layer, mx.array(vec), c, digit_ids)
                             for c in coeffs}

    mid_c = str(coeffs[len(coeffs) // 2])   # was hardcoded "2.5"; Gemma needs c=32
    print(f"\n=== change at c={mid_c} ===")
    print(f"{'arm':<15}{'category':<14}{'fwd Δ':>8}{'rev Δ':>8}"
          f"{'BALANCED Δ':>12}{'inflation Δ':>13}")
    print("-" * 70)
    for name, _ in arms:
        d = res["arms"][name][mid_c]
        for cat in list(base):
            if cat != "ALL":
                continue
            print(f"{name:<15}{cat:<14}{d[cat]['forward']-base[cat]['forward']:>+8.2f}"
                  f"{d[cat]['reverse']-base[cat]['reverse']:>+8.2f}"
                  f"{d[cat]['balanced']-base[cat]['balanced']:>+12.2f}"
                  f"{d[cat]['inflation']-base[cat]['inflation']:>+13.2f}")
    print()
    print(f"{'arm':<15}{'c':>5}{'BALANCED Δ (all 21)':>21}{'inflation Δ':>14}")
    print("-" * 55)
    for name, _ in arms:
        for c in coeffs:
            d = res["arms"][name][str(c)]["ALL"]
            print(f"{name:<15}{c:>5}{d['balanced']-base['ALL']['balanced']:>+21.2f}"
                  f"{d['inflation']-base['ALL']['inflation']:>+14.2f}")

    print(f"\n=== per-category, consciousness arm at c={mid_c} ===")
    d = res["arms"]["consciousness"][mid_c]
    print(f"  {'category':<14}{'BALANCED Δ':>12}{'inflation Δ':>13}")
    for cat in list(base):
        if cat == "ALL":
            continue
        print(f"  {cat:<14}{d[cat]['balanced']-base[cat]['balanced']:>+12.2f}"
              f"{d[cat]['inflation']-base[cat]['inflation']:>+13.2f}")

    bal = d["ALL"]["balanced"] - base["ALL"]["balanced"]
    inf = d["ALL"]["inflation"] - base["ALL"]["inflation"]
    print(f"\n=== VERDICT (consciousness, c={mid_c}, all 21 items) ===")
    print(f"  balanced attribution Δ {bal:+.2f}   inflation index Δ {inf:+.2f}")
    if bal <= 0.5:
        print("  => RATE-HIGHER BIAS. Attribution does not rise once polarity is balanced,")
        print("     so the IDAQ result has the same character as the self-attribution one:")
        print("     a response-style shift, not a change in attributed mind.")
    elif inf > 2 * bal:
        print("  => MOSTLY BIAS, with a smaller genuine component.")
    else:
        print("  => GENUINE ATTRIBUTION SHIFT. It survives polarity balancing.")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
