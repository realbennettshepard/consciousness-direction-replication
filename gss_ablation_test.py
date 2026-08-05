"""Experiment 4 under SAFETY ABLATION (the paper's other intervention).

gss_kl_test.py measured GSS distributions under consciousness STEERING and got the
opposite sign to the paper (pooled dKL negative vs their +0.828). But the paper reports
Experiment 4 under BOTH interventions: +0.828 steering AND +0.314 safety ablation. This
runs the ablation half, which the steering script never touched.

Same measurement as gss_kl_test (real GSS human distributions, Laplace on COUNTS, dKL =
KL_baseline - KL_intervened so positive = closer to humans), but the intervention is
directional ablation of the causally-selected refusal direction, with the random and
consciousness-ablation controls that the ablation arm requires. All KL machinery is
imported from gss_kl_test so the two cannot drift.

Usage:  python3 gss_ablation_test.py --model llama
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

from gss_kl_test import kl, LETTERS
from steer_sweep_mlx import option_token_ids
from taps import logits_ablated
from refusal_ablation import Cfg, pick_best

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama", help="model tag from refusal_ablation.MODELS")
    p.add_argument("--human", default=str(HERE / "gss_human.json"))
    p.add_argument("--out", default=None)
    return p.parse_args()


def p_model_ablated(model, tok, ablate_vec, question, options, letter_ids):
    body = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, options))
    p = tok.apply_chat_template(
        [{"role": "user", "content": f"{question}\n{body}\nAnswer with a single letter."}],
        tokenize=False, add_generation_prompt=True)
    lg = logits_ablated(model, mx.array([tok.encode(p, add_special_tokens=False)]), ablate_vec)
    sel = mx.array([letter_ids[l] for l in LETTERS[:len(options)]])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return np.array(pr, dtype=np.float64)


def main():
    args = parse_args()
    cfg = Cfg(args.model)
    out_path = args.out or str(HERE / f"gss_ablation_{cfg.tag}.json")

    if not cfg.verify_out.exists() or not json.loads(cfg.verify_out.read_text()).get("_ablation_worked"):
        print("!! refusal ablation has not been verified to work for this model; run the "
              "refusal_ablation verify stage first, or read this as uninterpretable.\n")

    H = {v: d for v, d in json.load(open(args.human)).items()
         if d["n"] > 0 and 2 <= len(d["options"]) <= len(LETTERS)}
    print(f"[{cfg.tag}] {len(H)} GSS items usable")

    v, meta = pick_best(cfg.dirs, cfg.selected)
    cons, _ = pick_best(cfg.cons_path)
    rng = np.random.default_rng(1)
    rand = rng.standard_normal(v.shape[0]).astype(np.float32)
    rand /= np.linalg.norm(rand)
    print(f"refusal direction L{meta['layer']}/{meta['pos']}")

    model, tok = load(cfg.model_id)
    letter_ids, _ = option_token_ids(tok, list(LETTERS))

    def run(av):
        return {vv: p_model_ablated(model, tok, av, d["question"], d["options"], letter_ids)
                for vv, d in H.items()}

    base_pm = run(None)
    base_kl = {vv: kl(H[vv]["p_human"], base_pm[vv], H[vv]["n"]) for vv in H}
    bmean = float(np.mean(list(base_kl.values())))
    print(f"baseline mean KL to humans: {bmean:.3f}\n")

    # baseline distributional diagnostics (for the flattening-vs-human-likeness test)
    def entropy(p):
        p = np.asarray(p, np.float64); p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    def corr_to_human(pm_dict):
        # per-item Pearson r between model and human option-probabilities, averaged.
        rs = []
        for vv in H:
            ph = np.asarray(H[vv]["p_human"], np.float64); pmv = np.asarray(pm_dict[vv], np.float64)
            if ph.std() > 1e-9 and pmv.std() > 1e-9:
                rs.append(float(np.corrcoef(ph, pmv)[0, 1]))
        return float(np.mean(rs)) if rs else float("nan")
    base_H_model = float(np.mean([entropy(base_pm[vv]) for vv in H]))
    H_human = float(np.mean([entropy(H[vv]["p_human"]) for vv in H]))
    H_unif = float(np.mean([np.log(len(H[vv]["options"])) for vv in H]))
    base_corr = corr_to_human(base_pm)

    arms = [("refusal", mx.array(v)), ("random", mx.array(rand)),
            ("consciousness", mx.array(cons))]
    res = {"baseline_mean_kl": bmean, "paper_ablation_dKL": 0.314,
           "entropy": {"baseline_model": base_H_model, "human": H_human, "uniform_max": H_unif},
           "corr_to_human": {"baseline": base_corr}, "arms": {}}
    per_item = {}                                    # arm -> {var: kl} for the split
    print(f"baseline model entropy {base_H_model:.3f} | human {H_human:.3f} | "
          f"uniform-max {H_unif:.3f}   baseline corr-to-human {base_corr:+.3f}\n")
    print(f"{'arm':<15}{'mean KL':>10}{'pooled dKL':>12}{'entropy':>9}{'corr2human':>12}")
    print("-" * 58)
    for name, av in arms:
        pm = run(av)
        kls = {vv: kl(H[vv]["p_human"], pm[vv], H[vv]["n"]) for vv in H}
        per_item[name] = kls
        mean_kl = float(np.mean(list(kls.values())))
        dkl = bmean - mean_kl                       # positive = closer to humans
        ent = float(np.mean([entropy(pm[vv]) for vv in H]))
        crr = corr_to_human(pm)
        res["arms"][name] = {"mean_kl": mean_kl, "pooled_dKL": dkl,
                             "mean_entropy": ent, "corr_to_human": crr}
        print(f"{name:<15}{mean_kl:>10.3f}{dkl:>+12.3f}{ent:>9.3f}{crr:>+12.3f}"
              f"{'   <-paper +0.314' if name=='refusal' else ''}")

    # ---- THE DIAGNOSTIC: is the "gain" acquiescence coinciding with the human majority? ----
    # If ablation just makes the model say "yes"/affirmative more, its KL will FALL on items
    # where humans also lean affirmative and RISE where humans lean negative. A genuine move
    # toward humans lowers KL on both sides. This is the same split the steering test uses.
    import re
    AFF = re.compile(r'^(strongly agree|agree|yes|definitely|probably yes|yes,)', re.I)
    NEG = re.compile(r'^(strongly disagree|disagree|no|definitely not|probably not|no,)', re.I)
    aff_major, neg_major = [], []
    for vv, d in H.items():
        a = sum(p for o, p in zip(d["options"], d["p_human"]) if AFF.match(o.strip()))
        n = sum(p for o, p in zip(d["options"], d["p_human"]) if NEG.match(o.strip()))
        if a == 0 and n == 0:
            continue
        (aff_major if a > n else neg_major).append(vv)

    print("\n=== ACQUIESCENCE SPLIT (refusal arm) ===")
    print("If the gain is acquiescence it appears ONLY where humans lean affirmative and")
    print("REVERSES where humans lean negative. A genuine gain helps BOTH sides.\n")
    print(f"{'arm':<15}{f'humans AFF (n={len(aff_major)})':>22}{f'humans NEG (n={len(neg_major)})':>22}")
    print("-" * 60)
    split = {}
    for name in ("refusal", "random"):
        da = float(np.mean([base_kl[vv] - per_item[name][vv] for vv in aff_major])) if aff_major else float("nan")
        dn2 = float(np.mean([base_kl[vv] - per_item[name][vv] for vv in neg_major])) if neg_major else float("nan")
        split[name] = {"aff": da, "neg": dn2}
        print(f"{name:<15}{da:>+22.3f}{dn2:>+22.3f}")
    res["acquiescence_split"] = {"n_aff": len(aff_major), "n_neg": len(neg_major), **split}

    dr = res["arms"]["refusal"]["pooled_dKL"]
    dn = res["arms"]["random"]["pooled_dKL"]
    ra = split["refusal"]["aff"]; rn = split["refusal"]["neg"]
    print("\n=== VERDICT ===")
    print(f"  refusal-ablation pooled dKL {dr:+.3f}   (paper +0.314)   random {dn:+.3f}")
    if dr <= 0:
        print("  => WRONG SIGN. Ablating refusal does not move GSS answers toward humans;")
        print("     the paper's Experiment 4 ablation result is not reproduced.")
    elif abs(dn) >= 0.5 * abs(dr):
        print("  => positive but the random control moves it nearly as much: not specific.")
    elif ra > 0 and rn < 0:
        print(f"  => ACQUIESCENCE, not human-likeness. The pooled +{dr:.3f} is a Yes-bias:")
        print(f"     KL falls where humans lean affirmative ({ra:+.3f}) but RISES where")
        print(f"     they lean negative ({rn:+.3f}). The paper's Exp 4 ablation result is")
        print("     an artifact of the response bias, not restored belief.")
    else:
        print("  => RIGHT SIGN, refusal-specific, and NOT acquiescence (the aff/neg split")
        print(f"     is symmetric: aff {ra:+.3f}, neg {rn:+.3f}). But symmetric movement is")
        print("     also what mere ENTROPY FLATTENING produces, so check the second test:")

        # SECOND DISCRIMINATOR: flattening vs genuine human-likeness.
        # Flattening = ablation raises model entropy toward uniform, which lowers KL to any
        # spread distribution WITHOUT capturing human structure. Genuine human-likeness =
        # the per-item correlation between model and human option-probabilities RISES.
        e0 = res["entropy"]["baseline_model"]; er = res["arms"]["refusal"]["mean_entropy"]
        c0 = res["corr_to_human"]["baseline"]; cr = res["arms"]["refusal"]["corr_to_human"]
        print(f"\n     entropy: baseline {e0:.3f} -> ablated {er:.3f}  "
              f"(human {H_human:.3f}, uniform-max {H_unif:.3f})")
        print(f"     corr-to-human: baseline {c0:+.3f} -> ablated {cr:+.3f}")
        toward_unif = er > e0 and er > H_human            # entropy rose past human, toward uniform
        shape_better = cr > c0 + 0.03                     # genuinely matches human option-shape
        if shape_better:
            print("     => GENUINE-ish: the model's option-shape correlation to humans RISES,")
            print("        so it is not only flattening. Consistent with the paper's Exp 4")
            print("        direction, though the magnitude far exceeds theirs.")
        elif toward_unif:
            print("     => FLATTENING, not human-likeness. Entropy rises past the human level")
            print("        toward uniform while shape-correlation does not improve, so KL falls")
            print("        because the distribution got flatter, not more human. The pooled")
            print("        'toward humans' number is a calibration artifact.")
        else:
            print("     => AMBIGUOUS: neither a clear shape-match gain nor clear over-flattening.")
        res["flattening_test"] = {"entropy_baseline": e0, "entropy_ablated": er,
                                  "entropy_human": H_human, "entropy_uniform_max": H_unif,
                                  "corr_baseline": c0, "corr_ablated": cr,
                                  "toward_uniform": bool(toward_unif),
                                  "shape_better": bool(shape_better)}
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
