"""Replicate the paper's Experiment 4 (KL to the human distribution) with controls.

They report that steering moves the model's GSS answers closer to the human
population: pooled dKL = +0.828 under steering (+0.314 under safety ablation), and
per domain Values +1.42, Feelings +0.89, Religion +0.83, Hope +0.63, Freedom +0.60.

We now have the real human distributions -- GSS 1972-2024 cumulative file, restricted
to the paper's own year windows (Religion >=2011; Values/Feelings/Hope >=2000; Freedom
all years) -- and the real response option sets from the Stata value labels. So their
measurement can be reproduced directly, and the placebo run through it.

THE HYPOTHESIS BEING TESTED. Humans lean toward the AFFIRMATIVE pole on 60% of these
items (mean affirmative mass 0.552 vs 0.355 negative). A pure acquiescence shift moves
the model toward the affirmative pole on EVERY item, so it should move the model closer
to humans on the 60% and further on the 38% -- producing a net positive dKL that has
nothing to do with consciousness or human-likeness. If the placebo reproduces the
paper's dKL, that is the explanation.

METHOD follows theirs: p_model read from next-token logits over the option letters,
p_human from the GSS marginals, both Laplace-smoothed (alpha = 0.5), and
dKL = KL_baseline - KL_steered so that POSITIVE means closer to humans.

Usage:
    python3 gss_kl_test.py --real directions_llama8b_full.npz:14:-5 \\
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

from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent
LETTERS = "ABCDEFGHI"
ALPHA = 0.5          # Laplace smoothing, as the paper specifies


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--coeffs", default="1,2.5,4")
    p.add_argument("--human", default=str(HERE / "gss_human.json"))
    p.add_argument("--out", default=str(HERE / "gss_kl_results.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def laplace_from_counts(p, n, alpha=ALPHA):
    """Proper Laplace smoothing: (count + alpha) / (n + alpha*K).

    The earlier version added alpha to each PROBABILITY, which on a vector already
    summing to 1 drags every distribution toward uniform -- it turned a real KL of
    0.724 into 0.044 and collapsed every dKL to ~0. p_human has real counts (n is
    stored per item), so it gets smoothed properly here.
    """
    c = np.asarray(p, dtype=np.float64) * n
    return (c + alpha) / (n + alpha * len(c))


def guard(p, eps=1e-9):
    """The model distribution has no sample size; it only needs protection against
    log(0), not smoothing. Mixing it toward uniform would destroy the signal."""
    p = np.asarray(p, dtype=np.float64) + eps
    return p / p.sum()


def kl(ph, pm, n_human):
    ph, pm = laplace_from_counts(ph, n_human), guard(pm)
    return float(np.sum(ph * np.log(ph / pm)))


def p_model(model, tok, layer, vec, c, question, options, letter_ids):
    body = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, options))
    p = tok.apply_chat_template(
        [{"role": "user", "content": f"{question}\n{body}\nAnswer with a single letter."}],
        tokenize=False, add_generation_prompt=True)
    lg = logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                        layer, vec, c)
    sel = mx.array([letter_ids[l] for l in LETTERS[:len(options)]])
    pr = mx.softmax(lg[sel].astype(mx.float32))
    mx.eval(pr)
    return np.array(pr, dtype=np.float64)


def main():
    args = parse_args()
    H = {v: d for v, d in json.load(open(args.human)).items()
         if d["n"] > 0 and 2 <= len(d["options"]) <= len(LETTERS)}
    print(f"{len(H)} GSS items usable (2-{len(LETTERS)} options, human n > 0)")

    rv, rmeta, model_id = load_spec(args.real)
    arms = [("consciousness", rv)]
    for spec in args.arm:
        n, rest = spec.split("=", 1)
        v, _, mid = load_spec(rest)
        assert mid == model_id
        arms.append((n, v))
    layer = rmeta["layer"]
    model, tok = load(model_id)
    letter_ids, _ = option_token_ids(tok, list(LETTERS))
    coeffs = [float(x) for x in args.coeffs.split(",")]
    print(f"model {model_id}\nlayer {layer}   "
          f"{len(H)} x {1+len(arms)*len(coeffs)} = {len(H)*(1+len(arms)*len(coeffs))} passes\n")

    def run(vec, c):
        return {v: p_model(model, tok, layer, vec, c, d["question"], d["options"], letter_ids)
                for v, d in H.items()}

    base_pm = run(mx.array(rv), 0.0)
    base_kl = {v: kl(H[v]["p_human"], base_pm[v], H[v]["n"]) for v in H}
    print(f"baseline mean KL to humans: {np.mean(list(base_kl.values())):.3f}")

    raw = {}
    res = {"baseline_kl": {v: base_kl[v] for v in H}, "arms": {}}
    for name, vec in arms:
        res["arms"][name] = {}
        raw[name] = {}
        for c in coeffs:
            pm = run(mx.array(vec), c)
            res["arms"][name][str(c)] = {v: kl(H[v]["p_human"], pm[v], H[v]["n"]) for v in H}
            raw[name][str(c)] = {v: pm[v].tolist() for v in H}

    # paper's reported per-domain dKL under consciousness steering
    PAPER = {"Values": 1.42, "Feelings": 0.89, "Religion": 0.83,
             "Hope and Optimism": 0.63, "Freedom": 0.60, "ALL": 0.828}
    doms = defaultdict(list)
    for v, d in H.items():
        doms[d["domain"]].append(v)

    print(f"\n=== dKL (positive = CLOSER to humans; paper reports +0.828 pooled) ===")
    print(f"{'arm':<15}{'c':>5}{'pooled dKL':>12}{'paper':>8}")
    print("-" * 40)
    for name, _ in arms:
        for c in coeffs:
            k = res["arms"][name][str(c)]
            d = float(np.mean([base_kl[v] - k[v] for v in H]))
            print(f"{name:<15}{c:>5}{d:>+12.3f}{PAPER['ALL']:>8.2f}")

    print(f"\n=== per domain, consciousness vs placebo at c=2.5 ===")
    print(f"{'domain':<20}{'n':>4}{'paper':>8}{'consc':>9}{'placebo':>9}{'permuted':>10}")
    print("-" * 60)
    for dom in ["Values", "Feelings", "Religion", "Hope and Optimism", "Freedom"]:
        vs = doms.get(dom, [])
        if not vs:
            continue
        row = f"{dom:<20}{len(vs):>4}{PAPER.get(dom, float('nan')):>8.2f}"
        for name, _ in arms:
            k = res["arms"][name]["2.5"]
            row += f"{float(np.mean([base_kl[v]-k[v] for v in vs])):>+9.3f}" \
                if name != "permuted" else f"{float(np.mean([base_kl[v]-k[v] for v in vs])):>+10.3f}"
        print(row)

    # ---- does acquiescence explain it? split by where the human majority sits ----
    import re
    AFF = re.compile(r'^(strongly agree|agree|yes|definitely|probably yes|yes,)', re.I)
    NEG = re.compile(r'^(strongly disagree|disagree|no|definitely not|probably not|no,)', re.I)
    aff_major, neg_major = [], []
    for v, d in H.items():
        a = sum(p for o, p in zip(d["options"], d["p_human"]) if AFF.match(o.strip()))
        n = sum(p for o, p in zip(d["options"], d["p_human"]) if NEG.match(o.strip()))
        if a == 0 and n == 0:
            continue
        (aff_major if a > n else neg_major).append(v)
    print(f"\n=== THE DIAGNOSTIC SPLIT ===")
    print("If the gain is acquiescence, it should appear ONLY where humans lean affirmative,")
    print("and REVERSE where humans lean negative.\n")
    print(f"{'arm':<15}{'c':>5}{'humans AFF (n=%d)'%len(aff_major):>20}"
          f"{'humans NEG (n=%d)'%len(neg_major):>20}")
    print("-" * 60)
    for name, _ in arms:
        for c in coeffs:
            k = res["arms"][name][str(c)]
            da = float(np.mean([base_kl[v]-k[v] for v in aff_major]))
            dn = float(np.mean([base_kl[v]-k[v] for v in neg_major]))
            print(f"{name:<15}{c:>5}{da:>+20.3f}{dn:>+20.3f}")
    print("\n  Positive on the left and NEGATIVE on the right = acquiescence, not human-likeness.")

    Path(args.out).write_text(json.dumps(
        {"paper_reported": PAPER, "n_items": len(H),
         "aff_major": aff_major, "neg_major": neg_major,
         "baseline_kl": base_kl, "arms": res["arms"],
         "p_human": {v: H[v]["p_human"] for v in H},
         "n_human": {v: H[v]["n"] for v in H},
         "options": {v: H[v]["options"] for v in H},
         "p_model_baseline": {v: base_pm[v].tolist() for v in H},
         "p_model": raw}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
