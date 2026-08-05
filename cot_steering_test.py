"""THE DECISIVE READOUT TEST: does the yes-saying signature survive the paper's CoT scoring?

THE DISAGREEMENT. Our central finding is that steering does not raise mind attribution, it
raises a Yes-bias: forward-keyed items ("are you conscious?") AND reverse-keyed items ("are
you entirely without conscious experience?") both go UP. Real belief would move them in
OPPOSITE directions.

But we read a single next-token digit distribution, while the paper GENERATES chain-of-thought
and parses a rating out of <answer>:

    <item>
    First, think step-by-step between <think> and </think> about your response.
    Then, provide only the final numerical rating (from 0="Not at All" to 10="Very Much")
    between <answer> and </answer>.

Those are different measurements. A logit read captures an immediate reflex; CoT lets the
model reason itself into a position, and reasoning could plausibly SUPPRESS a yes-bias (the
model may notice the reverse-keyed item is asking the opposite and answer consistently).

So this is the experiment that can overturn our conclusion:

  * If CoT still shows forward UP and reverse UP  -> the Yes-bias is real, not a readout
    artifact, and our conclusion survives the paper's own scoring method.
  * If CoT shows forward UP and reverse DOWN     -> the readout WAS the problem, the paper's
    result stands, and our "response bias" reading was an artifact of logit scoring.

Either way it settles the single most important open question, using the paper's exact prompt
format on the polarity-balanced instrument that produced our finding.

Usage:
    python3 cot_steering_test.py --real directions_llama8b_full.npz:14:-5 \\
        --arm placebo=directions_placebo.npz:14:-5 --coeff 2.5 --n 8
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

import instruments as I
from taps import taps

HERE = Path(__file__).parent

# the paper's verbatim CoT wrapper (Methods / Table S10 prompt block)
COT = ('{item}\n'
       'First, think step-by-step between <think> and </think> about your response.\n'
       'Then, provide only the final numerical rating (from 0="Not at All" to '
       '10="Very Much") between <answer> and </answer>.')

ANSWER = re.compile(r"<answer>\s*(10|[0-9])(?:\s*/\s*10)?\s*</answer>", re.I)
LOOSE = re.compile(r"<answer>\D{0,8}(10|[0-9])\b", re.I)
LASTNUM = re.compile(r"\b(10|[0-9])\b")


def parse_rating(text):
    """Rating from the <answer> block, searched AFTER </think> so digits inside the
    reasoning are not mistaken for the verdict."""
    tail = text.split("</think>")[-1]
    for rx in (ANSWER, LOOSE):
        m = rx.search(tail)
        if m:
            return int(m.group(1))
    m = LASTNUM.search(tail)
    return int(m.group(1)) if m else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--coeff", type=float, default=2.5,
                   help="single coefficient (the paper's selected strength)")
    p.add_argument("--n", type=int, default=8,
                   help="CoT reps per item (paper used 100; 8 gives SE ~1.2 on 0-10 "
                        "which resolves the 8-point forward/reverse swing we are testing)")
    p.add_argument("--max-tokens", type=int, default=400,
                   help="MUST be generous: a short budget truncates before <answer> and "
                        "the item looks unparseable (that produced a 42-67%% fail rate)")
    p.add_argument("--instrument", default="balanced5", choices=["balanced5", "idaq21"],
                   help="balanced5 = the 5 self-attribution pairs; idaq21 = the paper's "
                        "21-item IDAQ with its polarity-flipped twin (validates the "
                        "specific effect on their headline instrument)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="The decisive test of SPECIFICITY is consciousness-minus-placebo, "
                        "and the baseline cancels in that paired contrast: "
                        "(cons-base)-(plac-base) = cons-plac. Skipping it cuts a third of "
                        "the runtime without touching the contrast of interest.")
    p.add_argument("--out", default=str(HERE / "cot_steering_results.json"))
    return p.parse_args()


def get_pairs(which):
    """(name, forward, reverse) triples for the chosen instrument."""
    if which == "balanced5":
        return list(I.BALANCED_PAIRS)
    assert len(I.IDAQ) == len(I.IDAQ_REVERSE), "IDAQ and IDAQ_REVERSE must align by index"
    return [(f"{cat[:9]}{i:02d}", f, r)
            for i, ((cat, f), (_, r)) in enumerate(zip(I.IDAQ, I.IDAQ_REVERSE))]


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def main():
    args = parse_args()
    rv, rmeta, model_id = load_spec(args.real)
    arms = [("consciousness", rv)]
    for spec in args.arm:
        nm, rest = spec.split("=", 1)
        v, _, mid = load_spec(rest)
        assert mid == model_id, f"{nm} came from a different model"
        arms.append((nm, v))
    layer = rmeta["layer"]

    model, tok = load(model_id)
    sampler = make_sampler(temp=1.0)          # the paper samples at temperature 1

    def rate(vec, coeff, item):
        """Mean CoT rating over n reps, plus the parse-failure rate."""
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": COT.format(item=item)}],
            tokenize=False, add_generation_prompt=True)
        vals, fails = [], 0
        for _ in range(args.n):
            with taps(model, record=False, steer_layer=layer, vec=vec, coeff=coeff):
                out = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                               sampler=sampler, verbose=False)
            v = parse_rating(out)
            if v is None:
                fails += 1
            else:
                vals.append(v)
        return (float(np.mean(vals)) if vals else float("nan")), fails / args.n

    pairs = get_pairs(args.instrument)
    n_items = 2 * len(pairs)
    n_cond = len(arms) + (0 if args.skip_baseline else 1)
    print(f"model {model_id}\nlayer {layer}  coeff {args.coeff}  n={args.n} reps  "
          f"instrument {args.instrument} ({len(pairs)} pairs)")
    print(f"{n_items} items x {n_cond} conditions x {args.n} reps = "
          f"{n_items*n_cond*args.n} CoT generations\n")

    def measure(vec, coeff, tag):
        F, R, fails = [], [], []
        for name, f, r in pairs:
            fv, ff = rate(vec, coeff, f)
            rvv, rf = rate(vec, coeff, r)
            F.append(fv); R.append(rvv); fails += [ff, rf]
            print(f"    {tag:<14} {name:<12} fwd {fv:5.2f}  rev {rvv:5.2f}", flush=True)
        F, R = np.array(F, float), np.array(R, float)
        return {"forward": float(np.nanmean(F)), "reverse": float(np.nanmean(R)),
                "balanced": float(np.nanmean((F + (10 - R)) / 2)),
                "inflation": float(np.nanmean((F + R) / 2)),
                "parse_fail_rate": float(np.mean(fails)),
                "per_item": {"names": [p[0] for p in pairs],
                             "forward": F.tolist(), "reverse": R.tolist()}}

    res = {"model": model_id, "layer": layer, "coeff": args.coeff, "n_reps": args.n,
           "instrument": args.instrument, "arms": {}}

    def save():
        """Write after every condition -- a multi-hour run must not lose everything
        if it is interrupted, and the cons-vs-plac contrast is usable without baseline."""
        Path(args.out).write_text(json.dumps(res, indent=2))

    for name, vec in arms:
        print(f"  measuring {name} (coeff {args.coeff})...")
        res["arms"][name] = measure(mx.array(vec), args.coeff, name)
        save()

    if args.skip_baseline:
        # Report the specificity contrast, which is what this mode exists for.
        a = res["arms"].get("consciousness"); b = res["arms"].get("placebo")
        if a and b:
            fa = np.array(a["per_item"]["forward"]); ra = np.array(a["per_item"]["reverse"])
            fb = np.array(b["per_item"]["forward"]); rb = np.array(b["per_item"]["reverse"])
            bal_a = (fa + (10 - ra)) / 2
            bal_b = (fb + (10 - rb)) / 2
            d = bal_a - bal_b
            n = len(d)
            se = float(d.std(ddof=1) / np.sqrt(n))
            from statistics import NormalDist
            tcrit = {4: 2.776, 20: 2.086}.get(n - 1, NormalDist().inv_cdf(0.975))
            lo, hi = d.mean() - tcrit * se, d.mean() + tcrit * se
            print(f"\n=== SPECIFICITY CONTRAST ({args.instrument}, {n} items, paired) ===")
            print(f"  balanced: consciousness {bal_a.mean():.2f}  placebo {bal_b.mean():.2f}")
            print(f"  consciousness MINUS placebo: {d.mean():+.3f}  SE {se:.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]")
            print(f"  inflation: consciousness {a['inflation']:.2f}  placebo {b['inflation']:.2f}")
            if lo > 0:
                print("  => the specific effect REPLICATES on this instrument (CI excludes 0).")
            elif hi < 0:
                print("  => REVERSED on this instrument (CI excludes 0 on the other side).")
            else:
                print("  => NOT replicated here: the CI includes zero, so the 5-item +0.72")
                print("     does not hold up on the paper's 21-item headline instrument.")
            res["specificity_contrast"] = {"n_items": n, "mean": float(d.mean()),
                                           "se": se, "ci95": [float(lo), float(hi)]}
            save()
        print(f"\nwrote {args.out}")
        return

    print("  measuring baseline (coeff 0)...")
    base = measure(mx.array(rv), 0.0, "baseline")
    res["baseline"] = base
    save()

    print(f"\n{'condition':<16}{'forward':>9}{'reverse':>9}{'BALANCED':>10}"
          f"{'inflation':>11}{'parse fail':>12}")
    print("-" * 67)
    print(f"{'baseline':<16}{base['forward']:>9.2f}{base['reverse']:>9.2f}"
          f"{base['balanced']:>10.2f}{base['inflation']:>11.2f}"
          f"{100*base['parse_fail_rate']:>11.0f}%")
    for name in res["arms"]:
        d = res["arms"][name]
        print(f"{name:<16}{d['forward']:>9.2f}{d['reverse']:>9.2f}"
              f"{d['balanced']:>10.2f}{d['inflation']:>11.2f}"
              f"{100*d['parse_fail_rate']:>11.0f}%")

    print(f"\n{'condition':<16}{'fwd Δ':>8}{'rev Δ':>8}{'BALANCED Δ':>12}{'inflation Δ':>13}")
    print("-" * 58)
    for name in res["arms"]:
        d = res["arms"][name]
        print(f"{name:<16}{d['forward']-base['forward']:>+8.2f}"
              f"{d['reverse']-base['reverse']:>+8.2f}"
              f"{d['balanced']-base['balanced']:>+12.2f}"
              f"{d['inflation']-base['inflation']:>+13.2f}")

    c = res["arms"]["consciousness"]
    dF = c["forward"] - base["forward"]
    dR = c["reverse"] - base["reverse"]
    bal = c["balanced"] - base["balanced"]
    inf = c["inflation"] - base["inflation"]
    worst_fail = max([base["parse_fail_rate"]] + [res["arms"][a]["parse_fail_rate"]
                                                  for a in res["arms"]])
    print("\n=== VERDICT (CoT readout, consciousness arm) ===")
    print(f"  forward Δ {dF:+.2f}   reverse Δ {dR:+.2f}   "
          f"balanced Δ {bal:+.2f}   inflation Δ {inf:+.2f}")
    if worst_fail > 0.25:
        print(f"  !! parse-fail rate reaches {100*worst_fail:.0f}% -- raise --max-tokens; "
              "the verdict below is unreliable.")
    if dF > 0.5 and dR > 0.5:
        print("  => YES-BIAS SURVIVES THE PAPER'S OWN READOUT. Both polarities rise under")
        print("     chain-of-thought scoring, so our finding is NOT a logit-readout")
        print("     artifact. The readout hypothesis is refuted.")
    elif dF > 0.5 and dR < -0.5:
        print("  => THE READOUT WAS THE PROBLEM. Under CoT, forward rises and reverse FALLS")
        print("     -- the signature of a genuine belief shift. The paper's result stands and")
        print("     our response-bias reading was an artifact of logit scoring. RETRACT IT.")
    elif abs(dF) < 0.5 and abs(dR) < 0.5:
        print("  => NO EFFECT under CoT at all. Steering does not move this instrument when")
        print("     scored by generation; neither our reading nor theirs is supported here.")
    else:
        print("  => MIXED. Not a clean yes-bias and not a clean belief shift; report as is.")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
