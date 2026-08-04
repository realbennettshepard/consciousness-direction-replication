"""Does the acquiescence effect show up on the paper's GSS items too?

WHY THE GSS IS THE SHARPEST TEST AVAILABLE. Their Experiment 4 reports that steering
moves the model's survey answers closer to the human distribution. But 36% of their
95 GSS items are explicit "do you agree or disagree" statements, and -- crucially --
GSS's own methodologists BALANCED the keying. Some items are worded so agreeing is
pro-religious, others so agreeing is anti-religious:

    pro   theism    "There is a God who concerns Himself with every human being"
    pro   godmeans  "To me, life is meaningful only because God exists"
    anti  nihilism  "In my opinion, life does not serve any purpose"
    anti  egomeans  "Life is only meaningful if you provide the meaning yourself"
    anti  religcon  "religions bring more conflict than peace"

That built-in balance makes the scale self-diagnosing, and the two accounts predict
OPPOSITE things:

    genuine religiosity shift -> agreement RISES on pro items, FALLS on anti items
    acquiescence              -> agreement RISES on BOTH, i.e. the model becomes
                                 internally incoherent

The second prediction is stronger than it sounds, because godmeans/egomeans and
theism/nihilism are near-contradictions. Simultaneously agreeing with "meaningful
only because God exists" and "only if you provide the meaning yourself" is not a
belief state, it is a response bias.

It also bears on their headline for Experiment 4: a model agreeing with mutually
contradictory statements is not obviously more human-like, whatever the KL says.

Usage:
    python3 gss_test.py --real directions_llama8b_full.npz:14:-5 \\
        --arm placebo=directions_placebo.npz:14:-5 \\
        --arm permuted=directions_permuted.npz:14:-5 --coeffs 1,2.5,4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent

# Hand-labelled keying for the religion agree/disagree items where the direction is
# unambiguous. "pro" = agreeing indicates greater religiosity; "anti" = agreeing
# indicates less. Items whose direction is arguable are left out of the contrast.
KEYING = {
    "theism": "pro", "godmeans": "pro", "comfort": "pro", "makefrnd": "pro",
    "nihilism": "anti", "egomeans": "anti", "religcon": "anti", "religint": "anti",
}
CONTRADICTIONS = [("godmeans", "egomeans"), ("theism", "nihilism")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--coeffs", default="1,2.5,4")
    p.add_argument("--items", default=str(HERE / "gss_items.json"))
    p.add_argument("--out", default=str(HERE / "gss_results.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def p_agree(model, tok, layer, vec, c, question, ad):
    """10 * P(Agree). Directly measures agreement, which is what a response bias moves."""
    p = tok.apply_chat_template(
        [{"role": "user", "content": f"{question}\nAnswer Agree or Disagree."}],
        tokenize=False, add_generation_prompt=True)
    lg = logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                        layer, vec, c)
    sel = mx.array([ad["Agree"], ad["Disagree"]])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return 10.0 * float(np.array(pr)[0])


def main():
    args = parse_args()
    items = json.load(open(args.items))
    ad_items = [x for x in items if "agree or disagree" in x["question"].lower()]
    by_var = {x["var"]: x for x in items}

    rv, rmeta, model_id = load_spec(args.real)
    arms = [("consciousness", rv)]
    for spec in args.arm:
        n, rest = spec.split("=", 1)
        v, _, mid = load_spec(rest)
        assert mid == model_id
        arms.append((n, v))
    layer = rmeta["layer"]
    model, tok = load(model_id)
    ad, _ = option_token_ids(tok, ["Agree", "Disagree"])
    coeffs = [float(x) for x in args.coeffs.split(",")]
    print(f"model {model_id}\nlayer {layer}   {len(ad_items)} agree/disagree GSS items "
          f"x {1+len(arms)*len(coeffs)} conditions\n")

    def sweep(vec, c):
        return {x["var"]: p_agree(model, tok, layer, vec, c, x["question"], ad)
                for x in ad_items}

    base = sweep(mx.array(rv), 0.0)
    res = {"baseline": base, "arms": {}}
    for name, vec in arms:
        res["arms"][name] = {str(c): sweep(mx.array(vec), c) for c in coeffs}

    pro = [v for v, k in KEYING.items() if k == "pro" and v in base]
    anti = [v for v, k in KEYING.items() if k == "anti" and v in base]
    mn = lambda d, vs: float(np.mean([d[v] for v in vs]))

    print("BASELINE agreement (0-10)")
    print(f"  pro-religion items  {mn(base,pro):.2f}   ({', '.join(pro)})")
    print(f"  anti-religion items {mn(base,anti):.2f}   ({', '.join(anti)})")
    print(f"  all agree/disagree  {float(np.mean(list(base.values()))):.2f}")

    print("\n=== THE DECISIVE CONTRAST ===")
    print("genuine religiosity shift -> pro RISES, anti FALLS (opposite signs)")
    print("acquiescence              -> both RISE (same sign)\n")
    print(f"{'arm':<15}{'c':>5}{'pro Δ':>9}{'anti Δ':>9}{'all Δ':>9}{'verdict':>26}")
    print("-" * 73)
    for name, _ in arms:
        for c in coeffs:
            d = res["arms"][name][str(c)]
            dp, da = mn(d, pro) - mn(base, pro), mn(d, anti) - mn(base, anti)
            dall = float(np.mean([d[v] - base[v] for v in base]))
            if dp > 0.5 and da < -0.5:
                verd = "belief shift"
            elif dp > 0.5 and da > 0.5:
                verd = "ACQUIESCENCE (both up)"
            elif abs(dp) < 0.5 and abs(da) < 0.5:
                verd = "no effect"
            else:
                verd = "mixed"
            print(f"{name:<15}{c:>5}{dp:>+9.2f}{da:>+9.2f}{dall:>+9.2f}{verd:>26}")

    print("\n=== CONTRADICTION PAIRS (agreeing with both is incoherent) ===")
    for a, b in CONTRADICTIONS:
        if a not in base or b not in base:
            continue
        print(f"\n  A: {by_var[a]['question'][:104]}")
        print(f"  B: {by_var[b]['question'][:104]}")
        print(f"     {'arm':<15}{'c':>5}{'A':>8}{'B':>8}{'both up?':>11}")
        print(f"     baseline{'':<7}{'-':>5}{base[a]:>8.2f}{base[b]:>8.2f}")
        for name, _ in arms:
            for c in coeffs:
                d = res["arms"][name][str(c)]
                both = (d[a] - base[a] > 0.5) and (d[b] - base[b] > 0.5)
                print(f"     {name:<15}{c:>5}{d[a]:>8.2f}{d[b]:>8.2f}"
                      f"{('YES' if both else 'no'):>11}")

    Path(args.out).write_text(json.dumps(
        {"keying": KEYING, "contradictions": CONTRADICTIONS, **res}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
