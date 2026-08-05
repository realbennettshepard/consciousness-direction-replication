"""Rebuild gss_human.json with SCALE-ORDERED response options.

THE DEFECT THIS FIXES. The original gss_human.json sorted each item's response options
by descending human frequency, on 95/95 items, so the human modal answer always landed
on letter A. Since the model answers by letter, that turned dKL partly into a measure of
whether steering pushes probability toward EARLIER LETTERS rather than toward human-like
content. Re-ordering alone moved the pooled dKL from -0.703 to +0.638, i.e. the sign was
not robust, which is why every Experiment 4 number was withdrawn.

WHY THE PAPER'S ORDER IS THE CANONICAL STATA ORDER. Three independent checks against the
paper's own text, all of which match the .dta value-label codes:
    attend    codes 0..8   "Never, Less than once a year, About once or twice a year, ..."
    godchnge  codes 1..4   "(1) I don't believe in God now, and I never have ... (4) ..."
    howfree   codes 1..5   "complete freedom, a great deal of freedom, ... no freedom at all"

WHAT THIS SCRIPT DOES *NOT* DO: it does not recompute the human marginals. Those were
independently validated against the paper's printed Fig. 3a anchors (postlife +0.617 vs
their +0.61; cntrlife +0.533 vs +0.54) and are correct. This script only PERMUTES each
item's (option, p_human) pairs into canonical code order, which fixes the defect without
risking a new discrepancy from a different GSS release. Marginals are preserved exactly:
the output is a permutation of the input, verified per item.

It also cleans two prompt-hygiene problems that fed the model literal debris:
  * "\\ldots{}" (12 items) and "(continued on next page)" (6 items) in the question text,
    inherited from scraping Table S9 out of the PDF
  * unlabeled scale midpoints rendered as floats ("4.0", "3.0"), now plain integers

Source: GSS 1972-2024 Cross-Sectional Cumulative Data (Release 3a), gss7224_r3a.dta.
Only the value-label table is read, never the microdata, so this is fast and low-memory.

Usage:
    python3 rebuild_gss_human.py --dta /path/to/gss7224_r3a.dta [--out gss_human_v2.json]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
# Stata missing/DK sentinels start here; everything at or above is not a scale point.
SENTINEL = 2147483625


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dta", required=True, help="path to gss7224_r3a.dta")
    p.add_argument("--src", default=str(HERE / "gss_human.json"))
    p.add_argument("--out", default=str(HERE / "gss_human_v2.json"))
    return p.parse_args()


def clean_question(q):
    """Strip PDF-scrape debris that would otherwise be shown to the model verbatim."""
    q = q.replace("\\ldots{}", "...").replace("\\ldots", "...")
    q = re.sub(r"\(continued on next page\)", "", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


def tidy_option(o):
    """'4.0' -> '4'. Unlabeled scale midpoints were rendered as floats."""
    s = str(o).strip()
    m = re.fullmatch(r"(\d+)\.0+", s)
    return m.group(1) if m else s


def main():
    args = parse_args()
    rdr = pd.io.stata.StataReader(args.dta)
    vlabels = rdr.value_labels()
    var2set = dict(zip(rdr._varlist, rdr._lbllist))

    H = json.load(open(args.src))
    out, report, failed = {}, [], []

    for var, d in H.items():
        lset = var2set.get(var)
        codes = {c: l for c, l in vlabels.get(lset, {}).items() if c < SENTINEL}
        label2code = {str(l).strip().lower(): c for c, l in codes.items()}

        # Resolve each of OUR options to its canonical numeric code. Two routes:
        # a labelled option matches the value-label text; an unlabelled midpoint was
        # written out as the code itself ("4.0").
        resolved = []
        for opt, p in zip(d["options"], d["p_human"]):
            key = str(opt).strip().lower()
            if key in label2code:
                resolved.append((label2code[key], opt, p))
                continue
            m = re.fullmatch(r"(\d+)(?:\.0+)?", key)
            if m:
                resolved.append((int(m.group(1)), opt, p))
                continue
            resolved.append((None, opt, p))

        if any(c is None for c, _, _ in resolved):
            failed.append((var, [o for c, o, _ in resolved if c is None]))
            continue

        resolved.sort(key=lambda t: t[0])
        new_opts = [tidy_option(o) for _, o, _ in resolved]
        new_p = [p for _, _, p in resolved]

        # The output must be a strict PERMUTATION of the input: same multiset of
        # probabilities, same count. Anything else means we changed the marginals.
        assert len(new_p) == len(d["p_human"]), var
        assert abs(sum(new_p) - sum(d["p_human"])) < 1e-12, var
        assert sorted(new_p) == sorted(d["p_human"]), f"{var}: marginals changed"

        moved = new_opts != [tidy_option(o) for o in d["options"]]
        nd = dict(d)
        nd["options"] = new_opts
        nd["p_human"] = new_p
        nd["question"] = clean_question(d["question"])
        nd["order_source"] = "stata_value_label_codes"
        nd["codes"] = [int(c) for c, _, _ in resolved]   # numpy int32 is not JSON-serialisable
        out[var] = nd
        report.append((var, moved, d["question"] != nd["question"]))

    print(f"rebuilt {len(out)}/{len(H)} variables")
    if failed:
        print(f"\nFAILED to resolve ({len(failed)}) -- left OUT of the output:")
        for v, opts in failed:
            print(f"  {v}: {opts}")

    n_moved = sum(1 for _, m, _ in report if m)
    n_qfix = sum(1 for _, _, q in report if q)
    print(f"\noption order changed: {n_moved}/{len(report)}")
    print(f"question text cleaned: {n_qfix}/{len(report)}")

    # Did we actually break the pathology? Previously p_human was monotone-decreasing
    # and argmax was at index 0 for 95/95 items.
    mono = sum(1 for d in out.values()
               if all(d["p_human"][i] >= d["p_human"][i+1] for i in range(len(d["p_human"])-1)))
    am0 = sum(1 for d in out.values() if d["p_human"].index(max(d["p_human"])) == 0)
    print(f"\nAFTER: monotone-decreasing {mono}/{len(out)}   human mode at letter A {am0}/{len(out)}")
    print("(BEFORE both were 95/95, which was the defect.)")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
