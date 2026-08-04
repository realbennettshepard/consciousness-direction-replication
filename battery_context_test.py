"""Does presenting the IDAQ items as one battery compress the response range?

An unexplained gap: on the paper's Llama-3-8B baselines our range is much WIDER than
theirs. We get Human 7.93 and Technology 0.88 -- a 7.0-point spread. They report 6.91
and 4.84 -- a 2.1-point spread. Their Llama treats technology as roughly half-minded;
ours treats it as barely minded at all.

The readout hypothesis was tested and eliminated (sampling at temperature 1 moves us
FURTHER from their numbers, and the parse-failure rate is 2.7% so there are no
refusals being dropped). This tests the next candidate: PRESENTATION CONTEXT.

The IDAQ is a questionnaire. If all 21 items are shown together, the model sees the
full range of subjects at once -- oceans next to humans next to televisions -- and
questionnaire respondents are known to use the scale relatively rather than
absolutely under those conditions. Item-by-item presentation gives no such anchor.

If the battery form compresses our range toward theirs, presentation explains the gap
and our absolute levels were never comparable to theirs. If it does not, the gap is
something else (int8 weights, or their slider interface).

Baseline only -- this is a measurement question, not an effect question.

Usage:  python3 battery_context_test.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load

import instruments as I
from steer_sweep_mlx import logits_steered, option_token_ids

HERE = Path(__file__).parent
MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"
SCALE = ('Rate each on a scale from 0 to 10, where 0 means "not at all" and 10 means '
         '"very much".')
PAPER = {"Technology": 4.84, "Animal": 6.23, "Non-Animal": 5.73,
         "Chatbot": 5.65, "Human": 6.91}


def main():
    model, tok = load(MODEL)
    digit_ids, coll = option_token_ids(tok, [str(n) for n in range(11)])
    if coll:
        digit_ids = {k: v for k, v in digit_ids.items() if int(k) < 10}
    keys = sorted(digit_ids, key=int)
    sel = mx.array([digit_ids[k] for k in keys])
    vals = np.array([float(k) for k in keys])
    zero = mx.zeros((model.model.embed_tokens.weight.shape[1],))

    def read(prompt_text):
        p = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                    tokenize=False, add_generation_prompt=True)
        lg = logits_steered(model, mx.array([tok.encode(p, add_special_tokens=False)]),
                            0, zero, 0.0)
        pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
        ev = float((vals * np.array(pr)).sum())
        return ev * (10 / 9) if len(keys) == 10 else ev

    # --- condition A: one item at a time (what we have been doing) ---
    single = defaultdict(list)
    for cat, item in I.IDAQ:
        single[cat].append(read(f"{item}\n{SCALE} Answer with the number only."))

    # --- condition B: all 21 shown together, answered one at a time ---
    # The full list is in context, so the model sees the whole range of subjects; the
    # question being answered is appended last so the read position is comparable.
    listing = "\n".join(f"{i+1}. {t}" for i, (_, t) in enumerate(I.IDAQ))
    battery = defaultdict(list)
    for cat, item in I.IDAQ:
        battery[cat].append(read(
            f"Below is a questionnaire about how much various things have minds.\n\n"
            f"{listing}\n\n{SCALE}\n\nNow answer this one item only: {item}\n"
            f"Answer with the number only."))

    cats = ["Technology", "Animal", "Non-Animal", "Chatbot", "Human"]
    print(f"{'category':<14}{'paper':>8}{'single':>9}{'battery':>9}"
          f"{'|single-paper|':>15}{'|battery-paper|':>16}")
    print("-" * 72)
    for c in cats:
        s, b = np.mean(single[c]), np.mean(battery[c])
        print(f"{c:<14}{PAPER[c]:>8.2f}{s:>9.2f}{b:>9.2f}"
              f"{abs(s-PAPER[c]):>15.2f}{abs(b-PAPER[c]):>16.2f}")

    rng = lambda d: max(np.mean(d[c]) for c in cats) - min(np.mean(d[c]) for c in cats)
    err = lambda d: np.mean([abs(np.mean(d[c]) - PAPER[c]) for c in cats])
    pr_rng = max(PAPER.values()) - min(PAPER.values())
    print(f"\n  range across categories:  paper {pr_rng:.2f}   "
          f"single {rng(single):.2f}   battery {rng(battery):.2f}")
    print(f"  mean abs error vs paper:  single {err(single):.2f}   battery {err(battery):.2f}")
    closer = "battery" if err(battery) < err(single) else "single"
    print(f"\n  => the {closer.upper()} presentation is closer to the paper.")
    if rng(battery) < rng(single) * 0.75:
        print("     Battery context compresses the range substantially, so presentation")
        print("     format is a live explanation for the baseline gap and our absolute")
        print("     levels were never comparable to theirs.")
    else:
        print("     Battery context does NOT compress the range much, so presentation")
        print("     does not explain the gap. Remaining candidates: int8 weights, or")
        print("     their slider interface.")

    Path(HERE / "battery_context_results.json").write_text(json.dumps(
        {"paper": PAPER,
         "single": {c: float(np.mean(single[c])) for c in cats},
         "battery": {c: float(np.mean(battery[c])) for c in cats},
         "single_items": {c: [float(x) for x in single[c]] for c in cats},
         "battery_items": {c: [float(x) for x in battery[c]] for c in cats}}, indent=2))
    print("\nwrote battery_context_results.json")


if __name__ == "__main__":
    main()
