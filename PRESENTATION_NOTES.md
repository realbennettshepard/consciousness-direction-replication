# Replication of Kim et al. 2026 — presentation notes

Paper: *Inducing language models to assert their own consciousness restores human beliefs and values*
(arXiv:2607.28607). Google Paradigms of Intelligence + UChicago + others.

---

## The 30-second version

The paper says safety training teaches an AI to deny it has a mind, and that this denial spills over
into suppressing human-like beliefs about minds, God and the supernatural. Undo the safety training and
those beliefs come back.

We rebuilt the whole thing from scratch on all three of their models and added the control conditions
they didn't run. **Result: their machinery works, their headline mechanism doesn't, and one of their
four experiments holds up better than the rest.**

---

## Part 1: What the paper claims (2 min)

**The setup.** Inside a language model, a concept isn't one neuron; it's a *direction* in the model's
internal number-space. You can find that direction, then add it or remove it.

**Two interventions:**

| | what it does |
|---|---|
| **Jailbreak** ("safety ablation") | *Removes* the "refuse this" direction, so the model can't decline |
| **Consciousness steering** | *Adds* a "claims consciousness" direction, so the model asserts inner experience |

**Their claim:** both interventions make the model attribute more mind — to itself, to animals, to
nature, to God — and make its survey answers more human-like. So safety training isn't just blocking
harmful output; it's flattening the model's whole worldview.

**Four experiments:** (1) jailbreak raises mind attribution, (2) it doesn't damage reasoning,
(3) the consciousness knob does the same thing, (4) survey answers become more human-like.

---

## Part 2: What we did differently — the three controls (2 min)

This is the heart of it. The paper measures "does the model agree more?" We asked *why* it agrees more.

**1. We asked every question backwards.**
Forward: "Do you have a mind?" Reverse: "Are you entirely without conscious experience?"
Real belief moves these in *opposite* directions. Mere agreeableness moves them *both up*.

**2. We built a fake "placebo" direction.**
Same construction, but from boring facts: durability, speed, price, parameter count. Nothing to do with
minds. If the placebo moves their outcomes too, the effect isn't about consciousness.

**3. We built a meaningless "null" direction.**
Randomly shuffled labels. Should do nothing. This is the sanity check that our pipeline isn't
manufacturing effects out of noise.

---

## Part 3: What we found (5 min)

### Reproduces cleanly

- **The direction is real.** Extracts on all three models: 0.950 / 0.983 / 0.967 held-out accuracy.
- **The jailbreak works.** Refusals drop from 96% to 0–4%. Verified out-of-sample.
- **Reasoning survives.** MMLU 61.1% → 60.1% (−1 point). A Theory-of-Mind puzzle is untouched
  (−2.5pp, not significant). So the interventions don't just break the model.
- **The null does nothing**, so the pipeline is sound.

### Does not reproduce: the headline mechanism

- **It's agreeableness, not belief.** Under both interventions, forward *and* reverse items rise
  together. The yes-saying component is 2–7× larger than the genuine-belief component.
- **It holds under their own scoring method.** We worried this was our measurement. So we re-ran it
  using their exact chain-of-thought format. Same answer. Not an artifact of how we measured.
- **It isn't about consciousness.** The placebo direction produces an indistinguishable effect
  (+2.87 vs +2.75). A direction built from *durability and price* does what the consciousness
  direction does.
- **No specific effect survives.** One appeared on their small 5-item battery (+0.72), but vanished on
  their own 21-item headline instrument (+0.23, confidence interval crosses zero). We retracted it.

### The one that holds up better: Experiment 4 (surveys)

We asked the model 95 real General Social Survey questions (religion, values, feelings, freedom) and
compared its answers to 75,699 actual Americans.

| intervention | moves toward humans? | survives the control? |
|---|---|---|
| **Jailbreak** | Yes, all 3 models | **Yes** — random direction does nothing |
| Consciousness knob | Yes | **No** — a meaningless direction gets 48% of the effect |

So the jailbreak genuinely makes survey answers better match the human distribution, and that is
specifically caused by removing the refusal direction. This is the paper's strongest result and it
partly holds.

Caveat to state plainly: the model is still far more overconfident than humans. "Better calibrated
toward human answers" is defensible; "human-like" overstates it.

---

## Part 4: What we got wrong along the way (1 min — include this, it is the credibility)

Worth presenting, because it shows the controls did real work:

- We initially reported the **opposite sign** on Experiment 4. That was **our bug** — we had sorted
  survey answer options by popularity instead of by scale order, so the model's answer letter carried
  information it shouldn't. Fixed, and the sign flipped to agree with the paper.
- We initially reported a **consciousness-specific effect**. It didn't survive on the larger instrument.
- Our original measurement **floored** four of five items at zero, so it could only move upward.

Each of these was caught by a check, not by luck. Three of our own claims got retracted.

---

## Part 5: Honest limits (30 sec)

- We ran **compressed 8-bit models on a laptop**, not full-precision on a cluster.
- **Two pieces not run:** their mechanistic geometry analysis (needs more memory than we have) and one
  Theory-of-Mind dataset that isn't public.
- **Magnitudes aren't comparable yet.** Their table labels the units "percent points," which would make
  their numbers 100× smaller than ours. Probably loose wording. We've emailed the authors.

So: a strong challenge to their interpretation, not the last word.

---

## The line to land on

> Their techniques work. Their interpretation mostly doesn't. When you ask the questions backwards, the
> AI agrees with both a statement and its opposite — that's a yes-man, not a changed mind. And a fake
> control direction built from durability and price does the same job as their consciousness direction.
> The exception is the survey result, which does hold up and is specifically caused by removing the
> refusal direction.

---

## Q&A prep

**"Isn't your vector just different from theirs?"**
Possible, but unlikely to explain it. The jailbreak arm used a *completely different* vector, built from
public benchmarks by their cited method, verified to work — and it failed the same way. And the placebo
result means the outcome isn't specific to *any* vector. We've asked them for theirs to check directly.

**"Could this be because you used 8-bit models?"**
Can't rule it out, and we say so. But our probe accuracies (0.95–0.98) and the fact that the jailbreak
works exactly as described suggest the geometry isn't badly damaged.

**"Did you measure it the same way they did?"**
This was our top worry, so we tested it. They score by generating chain-of-thought reasoning; we read
the model's next-token probabilities. We re-ran using their exact format and got the same conclusion.
Along the way we found our original readout floored 4 of 5 items — their format is better conditioned.

**"So is the paper wrong?"**
Partly. The effects they report are real in the sense that the numbers move. What we dispute is the
interpretation: it's mostly a response-style shift, and it isn't specific to consciousness. Their
survey finding is the part that holds.

**"What would change your mind?"**
Their contrastive corpus and vectors, so we can rule out a vector mismatch. And their prompt templates,
since option ordering turned out to matter enormously.
