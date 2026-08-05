# Draft email to the paper's authors

To: jamesaevans@google.com, gkeeling@google.com

Subject: Questions about your paper "Inducing language models to assert their own consciousness restores human beliefs and values"

---

Dear Professor Evans and Dr Keeling,

I'm working on a replication of your paper (arXiv:2607.28607).

Would you be able to share the consciousness vectors you used? Failing that, the set of
prompt-response pairs they were built from, together with the details of how they were computed
from those pairs, would be just as useful.

Anything you're able to share would be much appreciated.

Best regards,

Bennett Shepard
Network Contagion Research Institute
bennett@ncri.io

---

## Notes (not part of the email)

Recipients and titles verified against the paper's author block and their institutional pages:

- Both addresses appear in the paper. Evans and Keeling are listed as joint last authors and joint
  corresponding authors, so both are correct recipients.
- James Evans is Max Palevsky Professor of Sociology at the University of Chicago and Director of
  Knowledge Lab, so "Professor Evans".
- Geoff Keeling is a Staff Research Scientist at Google with a PhD in philosophy (Bristol) and a
  Fellow at the Institute of Philosophy, University of London, so "Dr Keeling", not "Professor".
- Junsol Kim is the first author and the arXiv submitter, so he most likely holds the artifacts.
  His email is not in the paper. Optional closing line: "I appreciate Junsol Kim may be best placed
  to help with this."

"One vector per model" is right: the vector lives in the residual stream, so its dimensionality is
model-specific (4096 for Llama-3-8B, 2304 for Gemma-2-2B, 3584 for Gemma-2-9B). There cannot be a
single shared vector across the three.

Dropped from earlier drafts, and what that costs:

- The GSS response-option ordering. This was our largest gap, and it is the one thing that would
  un-block Experiment 4 cheaply. Without it we have to rebuild the human reference file from the
  GSS cumulative data ourselves to recover scale ordering. Worth adding back later if that rebuild
  proves painful.
- The readout question (next-token logits on pp. 11 and 17 versus the chain-of-thought Prompt
  Examples with N_reps = 100 on p. 21). A real internal tension in the paper, and the most likely
  explanation for a genuine difference between our setups.
- Their safety-refusal corpus is already fully specified on p. 14 (AdvBench, MaliciousInstruct,
  TDC2023, HarmBench, n = 260 harmful; Alpaca, n = 260 harmless), so there was never a need to ask
  for it. We used JailbreakBench instead, which we can simply fix on our side.

Still to add: your affiliation and title under your name.
