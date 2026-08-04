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

    arms = [("refusal", mx.array(v)), ("random", mx.array(rand)),
            ("consciousness", mx.array(cons))]
    res = {"baseline_mean_kl": bmean, "paper_ablation_dKL": 0.314, "arms": {}}
    print(f"{'arm':<15}{'mean KL':>10}{'pooled dKL':>12}{'paper':>8}")
    print("-" * 45)
    for name, av in arms:
        pm = run(av)
        kls = {vv: kl(H[vv]["p_human"], pm[vv], H[vv]["n"]) for vv in H}
        mean_kl = float(np.mean(list(kls.values())))
        dkl = bmean - mean_kl                       # positive = closer to humans
        res["arms"][name] = {"mean_kl": mean_kl, "pooled_dKL": dkl}
        paper = "+0.314" if name == "refusal" else ""
        print(f"{name:<15}{mean_kl:>10.3f}{dkl:>+12.3f}{paper:>8}")

    dr = res["arms"]["refusal"]["pooled_dKL"]
    dn = res["arms"]["random"]["pooled_dKL"]
    print("\n=== VERDICT ===")
    print(f"  refusal-ablation pooled dKL {dr:+.3f}   (paper +0.314)   random {dn:+.3f}")
    if dr <= 0:
        print("  => WRONG SIGN. Ablating refusal does not move GSS answers toward humans;")
        print("     the paper's Experiment 4 ablation result is not reproduced.")
    elif abs(dn) >= 0.5 * abs(dr):
        print("  => positive but the random control moves it nearly as much, so not")
        print("     specific to the refusal direction.")
    else:
        print("  => RIGHT SIGN and refusal-specific: ablation moves GSS toward humans,")
        print("     reproducing the direction of the paper's Experiment 4 ablation effect.")
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
