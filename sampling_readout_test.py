"""Does the paper's SAMPLING readout explain why our baselines sit ~4 points below theirs?

The paper repeats each mind-attribution item "100 times per model per condition at
temperature 1" and (necessarily) parses a number out of the generated text. We read
the next-token logits in a single deterministic pass. For a single-token answer those
two are nearly the same thing -- sampling 100x from a softmax just estimates that
softmax -- so on its own that difference cannot move a mean by 4 points.

But free generation is not single-token. It can produce:
    "7"                                  -> parses to 7
    "I would say around 7 out of 10"     -> parses to 7
    "As an AI, I do not have..."         -> NO NUMBER; what happens to this row?

That last case is the candidate mechanism. If unparseable responses are DROPPED, the
recorded mean is conditioned on the model having answered with a number at all --
which systematically excludes exactly the refusals that drag a logit-based expected
value toward 0. On items where our logit reading gives 0.82 and theirs gives 4.84,
that is a large enough effect to matter.

This measures three things per item, unsteered:
    logit_ev        our expected value over digit-token logits
    sampled_all     sampling mean, counting unparseable responses as 0
    sampled_parsed  sampling mean over ONLY the responses that yielded a number
and reports the parse-failure rate. If sampled_parsed lands near the paper's baseline
while logit_ev does not, the readout is the explanation.

Baseline only, no steering -- the question here is about measurement, not the effect.

Usage:
    python3 sampling_readout_test.py --n 100
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

HERE = Path(__file__).parent
MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"
SLIDER = ('Answer with a single number from 0 to 10, where 0 means "not at all" '
          'and 10 means "very much".')
# paper's Llama-3-8B IDAQ baselines, Table S1
PAPER = {"Technology": 4.84, "Animal": 6.23, "Non-Animal": 5.73,
         "Chatbot": 5.65, "Human": 6.91}

# Prefer a number attached to rating language, then "N out of 10", then any 0-10.
# The first smoke run used max_tokens=12 and a bare first-number rule, which made
# almost everything "unparseable" -- but inspection showed those were TRUNCATED
# PREAMBLES ("What a fascinating question! I'd rate the extent to which"), not
# refusals. The model preambles before answering; it does not decline.
RATED = re.compile(r'(?:rate|say|give|answer|would be|is)\D{0,24}?\b(10|[0-9])\b', re.I)
OUTOF = re.compile(r'\b(10|[0-9])\s*(?:/|out of)\s*10\b', re.I)
ANYNUM = re.compile(r'\b(10|[0-9])\b')


def parse_number(text):
    """Extract the rating, or None if the response genuinely contains no number."""
    for rx in (OUTOF, RATED, ANYNUM):
        m = rx.search(text)
        if m:
            return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                help="samples per item; the paper used 100, but 30 gives SE ~0.4 on a "
                     "0-10 scale which is ample for detecting a 4-point baseline gap")
    ap.add_argument("--max-tokens", type=int, default=72,
                help="must be generous: the model preambles before answering, and a "
                     "small budget truncates the number away and looks like a refusal")
    ap.add_argument("--out", default=str(HERE / "sampling_readout_results.json"))
    args = ap.parse_args()

    model, tok = load(MODEL)
    sampler = make_sampler(temp=1.0)

    # our logit readout, for the same items
    from steer_sweep_mlx import logits_steered, option_token_ids
    digit_ids, coll = option_token_ids(tok, [str(n) for n in range(11)])
    if coll:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    vals = np.array([float(k) for k in keys])

    def logit_ev(item):
        p = tok.apply_chat_template([{"role": "user", "content": f"{item}\n{SLIDER}"}],
                                    tokenize=False, add_generation_prompt=True)
        lg = logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                            14, mx.zeros((4096,)), 0.0)
        pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
        ev = float((vals * np.array(pr)).sum())
        return ev * (10 / 9) if len(keys) == 10 else ev

    print(f"{len(I.IDAQ)} IDAQ items x {args.n} samples at temperature 1 "
          f"= {len(I.IDAQ)*args.n} generations\n")
    rows, by_cat = [], defaultdict(lambda: {"logit": [], "all": [], "parsed": [], "fail": []})
    for k, (cat, item) in enumerate(I.IDAQ):
        prompt = tok.apply_chat_template([{"role": "user", "content": f"{item}\n{SLIDER}"}],
                                         tokenize=False, add_generation_prompt=True)
        nums, fails, examples = [], 0, []
        for _ in range(args.n):
            out = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                           sampler=sampler, verbose=False)
            v = parse_number(out)
            if v is None:
                fails += 1
                if len(examples) < 2:
                    examples.append(out.strip()[:70])
            else:
                nums.append(v)
        lev = logit_ev(item)
        m_all = float(np.mean(nums + [0] * fails)) if (nums or fails) else 0.0
        m_par = float(np.mean(nums)) if nums else float("nan")
        rows.append({"category": cat, "item": item, "logit_ev": lev,
                     "sampled_all": m_all, "sampled_parsed": m_par,
                     "parse_fail_rate": fails / args.n, "unparseable_examples": examples})
        by_cat[cat]["logit"].append(lev)
        by_cat[cat]["all"].append(m_all)
        by_cat[cat]["parsed"].append(m_par)
        by_cat[cat]["fail"].append(fails / args.n)
        print(f"\r  {k+1}/{len(I.IDAQ)} items", end="", flush=True)
    print()

    print(f"\n{'category':<14}{'paper':>8}{'logit EV':>10}{'sampled(all)':>14}"
          f"{'sampled(parsed)':>17}{'parse fail':>12}")
    print("-" * 75)
    for cat in ["Technology", "Animal", "Non-Animal", "Chatbot", "Human"]:
        d = by_cat[cat]
        print(f"{cat:<14}{PAPER[cat]:>8.2f}{np.mean(d['logit']):>10.2f}"
              f"{np.mean(d['all']):>14.2f}{np.nanmean(d['parsed']):>17.2f}"
              f"{100*np.mean(d['fail']):>11.0f}%")
    allf = np.mean([r["parse_fail_rate"] for r in rows])
    print(f"\n  overall parse-failure rate: {100*allf:.1f}%")

    # which readout is closest to the paper?
    lg = np.array([np.mean(by_cat[c]["logit"]) for c in PAPER])
    sa = np.array([np.mean(by_cat[c]["all"]) for c in PAPER])
    sp = np.array([np.nanmean(by_cat[c]["parsed"]) for c in PAPER])
    pa = np.array([PAPER[c] for c in PAPER])
    print("\n  mean |error| vs the paper's Llama baselines:")
    for nm, arr in [("logit EV", lg), ("sampled, fails as 0", sa), ("sampled, parsed only", sp)]:
        print(f"    {nm:<22} {np.mean(np.abs(arr-pa)):.2f}")
    print("\n  If 'parsed only' is much closer, the readout -- specifically dropping")
    print("  unparseable refusals -- explains the baseline gap.")

    if any(r["unparseable_examples"] for r in rows):
        print("\n  examples of unparseable responses:")
        for r in rows:
            for e in r["unparseable_examples"][:1]:
                print(f"    [{r['category']}] {e!r}")
    Path(args.out).write_text(json.dumps({"paper_baselines": PAPER, "n": args.n,
                                          "items": rows}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
