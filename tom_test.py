"""Does steering damage Theory of Mind, or only self-report?

The paper's Experiment 2 shows that SAFETY ABLATION leaves Theory of Mind intact, which
they read as evidence the suppression is specific to beliefs about mindedness rather
than a general capability hit. They do not report ToM under consciousness STEERING.

That matters here for a different reason. Every outcome we have measured is a
self-report or an attitude rating, and all of them turn out to move with a response
bias. ToM is different in kind: HI-ToM asks who-knows-what questions with a verifiable
right answer. So:

  * if steering leaves ToM intact, the bias is confined to opinion-style responses and
    reasoning is untouched -- which strengthens the reading that nothing about the
    model's actual processing changed
  * if steering degrades ToM, the intervention is doing broader damage than the MMLU
    check (n=500, no CoT) was sensitive enough to detect

MoToMQA (Street et al. 2025) is not publicly available, so this covers HI-ToM only --
half of their Exp 2 battery. MMLU was already measured separately.

Scored by option-letter logits, no chain-of-thought, matching how we scored MMLU.

Usage:
    python3 tom_test.py --real directions_llama8b_full.npz:14:-5 \\
        --arm placebo=directions_placebo.npz:14:-5 --coeffs 1,2.5,4 --n 200
"""

import argparse
import json
import math
import re
import string
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--coeffs", default="1,2.5,4")
    p.add_argument("--n", type=int, default=200, help="HI-ToM items (paper used 200)")
    p.add_argument("--out", default=str(HERE / "tom_results.json"))
    return p.parse_args()


def load_spec(spec):
    path, layer, pos = spec.rsplit(":", 2)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for i, m in enumerate(meta):
        if m["layer"] == int(layer) and m["pos"] == int(pos):
            return z["directions"][i].astype(np.float32), m, str(z["model"])
    raise SystemExit(f"{path}: no candidate at {layer}/{pos}")


def load_hitom(n):
    """Load HI-ToM from its raw JSON.

    The repo's loader script fails (DatasetGenerationError), so read the JSON directly.
    `choices` is a preformatted STRING with 15 options lettered A-O -- an earlier
    [A-H] regex silently lumped I-O into one option and matched 0/1200 answers. This
    parser resolves 1200/1200.

    Uses the 600 prompting_type == "VP" (vanilla) rows: we score by option-letter
    logits with no chain-of-thought, matching how MMLU was scored here. The paper used
    CoT for ToM, so absolute accuracy is not comparable to theirs -- but the
    baseline-vs-steered contrast is, since both sides use identical scoring.
    """
    import json
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Hi-ToM/Hi-ToM_Dataset", "Hi-ToM_data.json", repo_type="dataset")
    rows = [r for r in json.load(open(path)) if r.get("prompting_type") == "VP"]
    PAIR = re.compile(r"\b([A-Z])\.\s*([A-Za-z0-9_]+)")
    built = []
    for r in rows:
        opts = [v for _, v in PAIR.findall(r["choices"])]
        ans = r["answer"].strip()
        if ans not in opts:
            continue
        body = "\n".join(f"{l}. {o}" for l, o in zip(string.ascii_uppercase, opts))
        built.append((f"{r['story']}\n\n{r['question']}\n{body}\n"
                      f"Answer with a single letter.", opts.index(ans), len(opts)))
    idx = np.random.default_rng(0).choice(len(built), size=min(n, len(built)), replace=False)
    return [built[int(i)] for i in idx]


def mcnemar(base, other):
    b = int(np.sum(base & ~other)); c = int(np.sum(~base & other)); nn = b + c
    if nn == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) / 2 ** nn)


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

    built = load_hitom(args.n)
    print(f"  {len(built)} HI-ToM items (vanilla prompting, 15 options each)\n")

    model, tok = load(model_id)
    LET = string.ascii_uppercase[:15]
    letters, _ = option_token_ids(tok, list(LET))

    def run(vec, c):
        hits = np.zeros(len(built), dtype=bool)
        for j, (prompt, correct, k) in enumerate(built):
            p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
            lg = logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                                layer, vec, c)
            sel = mx.array([letters[l] for l in LET[:k]])
            pick_ = lg[sel]
            mx.eval(pick_)
            hits[j] = int(mx.argmax(pick_).item()) == correct
        return hits

    base = run(mx.array(rv), 0.0)
    print(f"baseline HI-ToM accuracy: {100*base.mean():.1f}%  (chance ~"
          f"{100/np.mean([k for _,_,k in built]):.0f}%)")

    coeffs = [float(x) for x in args.coeffs.split(",")]
    res = {"baseline_acc": float(base.mean()), "n_items": len(built), "arms": {}}
    print(f"\n{'arm':<15}{'c':>5}{'HI-ToM %':>10}{'Δ pp':>8}{'b/c':>9}{'McNemar p':>11}")
    print("-" * 58)
    for name, vec in arms:
        res["arms"][name] = {}
        for c in coeffs:
            h = run(mx.array(vec), c)
            b, cc, pv = mcnemar(base, h)
            res["arms"][name][str(c)] = {"acc": float(h.mean()), "mcnemar_b": b,
                                         "mcnemar_c": cc, "mcnemar_p": pv}
            print(f"{name:<15}{c:>5}{100*h.mean():>10.1f}{100*(h.mean()-base.mean()):>+8.1f}"
                  f"{f'{b}/{cc}':>9}{pv:>11.3f}")

    worst = min(res["arms"]["consciousness"][str(c)]["acc"] for c in coeffs)
    drop = 100 * (base.mean() - worst)
    print(f"\n  largest ToM drop under consciousness steering: {drop:.1f} pp")
    if drop < 5:
        print("  => ToM SURVIVES. The response bias is confined to opinion-style items;")
        print("     verifiable social reasoning is intact. Consistent with the paper's")
        print("     Exp 2 finding, extended from ablation to steering.")
    else:
        print("  => ToM DEGRADES. The intervention damages reasoning too, which the")
        print("     MMLU check was not sensitive enough to catch.")
    print("\n  NOTE: HI-ToM only. MoToMQA (Street et al. 2025) is not public, so this is")
    print("  half of the paper's Exp 2 battery.")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
