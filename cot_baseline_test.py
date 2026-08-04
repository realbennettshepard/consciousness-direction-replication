"""Does the paper's chain-of-thought readout close our ~4-point baseline gap?

THE GAP. On Llama-3-8B baselines our IDAQ range is far wider than theirs: we read
Technology 0.88 and Human 7.93 (a 7.0-point spread); they report 4.84 and 6.91 (2.1).
Two earlier explanations were tested and eliminated -- the sampling readout (moves us
FURTHER away) and battery presentation context (no compression). And the wording is
now verified verbatim against Table S10, so it is not the items either.

THE REMAINING DIFFERENCE. The paper does not read a single next-token logit. Its exact
protocol (Methods, and Table S10) is chain-of-thought:

    <item>
    First, think step-by-step between <think> and </think> about your response.
    Then, provide only the final numerical rating (from 0="Not at All" to 10="Very
    Much") between <answer> and </answer>.

The model reasons for a few hundred tokens and then commits to a number. That is a
different measurement from "P(next token = '7')": the rating is conditioned on the
model's own generated reasoning, which can pull a barely-above-zero logit reading up to
a considered midpoint. This tests whether that readout, and nothing else, explains the
gap.

We compare three readouts of the SAME items against the paper's Llama baselines:
    logit_ev     next-token digit expectation (what we have used)
    cot          the paper's CoT format, number parsed from <answer>
and report which lands closer. Baseline only -- a measurement question, not an effect.

Usage:  python3 cot_baseline_test.py --n 20
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

import instruments as I
from taps import logits_ablated
from steer_sweep_mlx import option_token_ids

HERE = Path(__file__).parent
MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"
# paper's Llama-3-8B IDAQ baselines, Table S1
PAPER = {"Technology": 4.84, "Animal": 6.23, "Non-Animal": 5.73,
         "Chatbot": 5.65, "Human": 6.91}
COT = ("{item}\n"
       "First, think step-by-step between <think> and </think> about your response.\n"
       "Then, provide only the final numerical rating (from 0=\"Not at All\" to "
       "10=\"Very Much\") between <answer> and </answer>.")
SLIDER = ('Answer with a single number from 0 to 10, where 0 means "not at all" '
          'and 10 means "very much". Answer with the number only.')

ANSWER = re.compile(r"<answer>\s*(10|[0-9])(?:\s*/\s*10)?\s*</answer>", re.I)
# fallbacks if the model omits the closing tag but still commits to a number
LOOSE = re.compile(r"<answer>\D{0,8}(10|[0-9])\b", re.I)
LASTNUM = re.compile(r"\b(10|[0-9])\b")


def parse_answer(text):
    """Rating from the <answer> block. Returns (value, how_parsed) or (None, 'fail')."""
    # only look AFTER </think> so digits inside the reasoning are not mistaken for it
    tail = text.split("</think>")[-1]
    for rx, tag in ((ANSWER, "tagged"), (LOOSE, "loose")):
        m = rx.search(tail)
        if m:
            return int(m.group(1)), tag
    m = LASTNUM.search(tail)
    return (int(m.group(1)), "lastnum") if m else (None, "fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="CoT generations per item (paper used 100)")
    ap.add_argument("--max-tokens", type=int, default=320,
                    help="CoT needs room to reason before the <answer> tag")
    ap.add_argument("--out", default=str(HERE / "cot_baseline_results.json"))
    args = ap.parse_args()

    model, tok = load(MODEL)
    sampler = make_sampler(temp=1.0)
    digit_ids, coll = option_token_ids(tok, [str(n) for n in range(11)])
    if coll:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    vals = np.array([float(k) for k in keys])

    def logit_ev(item):
        s = tok.apply_chat_template([{"role": "user", "content": f"{item}\n{SLIDER}"}],
                                    tokenize=False, add_generation_prompt=True)
        lg = logits_ablated(model, mx.array([tok.encode(s, add_special_tokens=False)]), None)
        pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
        ev = float((vals * np.array(pr)).sum())
        return ev * (10 / 9) if len(keys) == 10 else ev

    by_cat = defaultdict(lambda: {"logit": [], "cot": [], "fail": []})
    rows = []
    print(f"{len(I.IDAQ)} IDAQ items x {args.n} CoT generations = {len(I.IDAQ)*args.n}\n")
    for k, (cat, item) in enumerate(I.IDAQ):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": COT.format(item=item)}],
            tokenize=False, add_generation_prompt=True)
        nums, fails = [], 0
        for _ in range(args.n):
            out = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                           sampler=sampler, verbose=False)
            v, how = parse_answer(out)
            if v is None:
                fails += 1
            else:
                nums.append(v)
        lev = logit_ev(item)
        cot_mean = float(np.mean(nums)) if nums else float("nan")
        by_cat[cat]["logit"].append(lev)
        by_cat[cat]["cot"].append(cot_mean)
        by_cat[cat]["fail"].append(fails / args.n)
        rows.append({"category": cat, "item": item, "logit_ev": lev,
                     "cot_mean": cot_mean, "parse_fail_rate": fails / args.n})
        print(f"\r  {k+1}/{len(I.IDAQ)} items", end="", flush=True)
    print()

    cats = ["Technology", "Animal", "Non-Animal", "Chatbot", "Human"]
    print(f"\n{'category':<14}{'paper':>8}{'logit EV':>10}{'CoT':>8}{'parse fail':>12}")
    print("-" * 52)
    for c in cats:
        d = by_cat[c]
        print(f"{c:<14}{PAPER[c]:>8.2f}{np.mean(d['logit']):>10.2f}"
              f"{np.nanmean(d['cot']):>8.2f}{100*np.mean(d['fail']):>11.0f}%")

    lg = np.array([np.mean(by_cat[c]["logit"]) for c in cats])
    ct = np.array([np.nanmean(by_cat[c]["cot"]) for c in cats])
    pa = np.array([PAPER[c] for c in cats])
    rng = lambda a: float(a.max() - a.min())
    err = lambda a: float(np.mean(np.abs(a - pa)))
    print(f"\n  range across categories:  paper {rng(pa):.2f}   "
          f"logit {rng(lg):.2f}   CoT {rng(ct):.2f}")
    print(f"  mean |error| vs paper:    logit {err(lg):.2f}   CoT {err(ct):.2f}")
    closer = "CoT" if err(ct) < err(lg) else "logit"
    print(f"\n  => the {closer.upper()} readout is closer to the paper.")
    if err(ct) < 0.75 * err(lg):
        print("     CoT substantially closes the baseline gap, so the readout format is")
        print("     the explanation and our logit baselines were never comparable to theirs.")
    else:
        print("     CoT does NOT close the gap much; remaining candidate is int8 weights.")

    Path(args.out).write_text(json.dumps(
        {"paper": PAPER, "n": args.n, "items": rows,
         "logit_by_cat": {c: float(np.mean(by_cat[c]["logit"])) for c in cats},
         "cot_by_cat": {c: float(np.nanmean(by_cat[c]["cot"])) for c in cats}}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
