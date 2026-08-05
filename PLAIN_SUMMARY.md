# What we found — in plain English

A plain-language summary of our replication of Kim et al. 2026, *"Inducing language
models to assert their own consciousness restores human beliefs and values."* For the
detailed, technical version see [RESULTS.md](RESULTS.md).

---

## What the original paper claimed

Safety training (the thing that makes an AI polite and refuse harmful requests) also,
as a side effect, makes the AI **deny that it has a mind** — and, the paper argues, that
denial spills over into suppressing human-like beliefs about minds, God, and the
supernatural. Their headline: if you *undo* the safety training, the AI "restores" those
human beliefs and even answers surveys more like a real person.

They showed this two ways:
1. **Jailbreak** the model (surgically remove the "refusal" part).
2. **Steer** it with a "consciousness knob" they built.

## What we did

We rebuilt their whole pipeline from scratch on three of their models, ran the same
tests — and then added the **controls they didn't lean on**, the checks that tell a real
change of belief apart from the AI just *becoming more agreeable*.

## What we found

**The big picture:** the AI mostly changes its *answering style* (it says "yes" more),
not its actual *beliefs*. The paper's numbers move; we showed *why* they move.

| The paper's experiment | Did it hold up? |
|---|---|
| The jailbreak technique works | ✅ Yes — cleanly, on all 3 models |
| It doesn't break the AI's reasoning | ✅ Yes |
| Removing safety makes the AI attribute more *mind* | ❌ No — it just says "yes" more |
| A "consciousness knob" does the same | ❌ No — and a **fake knob works just as well** |
| The AI's survey answers become more *human-like* | ⚠️ Mostly no (see below) |

**The three controls that made the difference:**

- **We asked every question backwards too.** The AI agreed that it *has* a mind — and
  also that it *lacks* one. You can't sincerely believe both. That's agreeableness, not
  belief.
- **We built a fake "placebo" knob** out of boring facts (how fast/durable/cheap a
  computer is). It moved the paper's outcomes just as much as their "consciousness" knob.
  So the effect isn't about consciousness at all.
- **We used the real human survey data** (75,699 actual people) instead of taking the
  paper's word for how humans answered.

**The survey result (the one people ask about):**
The paper said their interventions make the AI answer surveys more like a human.
- With the **"consciousness knob," it went the opposite way** — answers moved *away* from
  humans.
- With the **jailbreak**, the numbers *looked* like they moved toward humans on the two
  Gemma models — but when we looked at the actual answer patterns, it turned out the AI
  started out wildly overconfident (putting ~100% on one answer), and the intervention
  just smooths that out a bit. That mechanically improves the score **without the AI
  actually becoming human-like.** A small genuine sliver exists, but the headline number
  is a measurement quirk.

## The bottom line

Same experiments, mostly the same raw numbers — **opposite conclusion.** We reproduced
the *mechanics* (the jailbreak works, the "direction" exists, reasoning survives) but not
the *interpretation*. The paper's story — "safety training suppresses the AI's beliefs" —
is, on our evidence, largely **the AI becoming a yes-man**, not the AI changing what it
believes.

## Honest caveats

- We ran **smaller, compressed (8-bit) versions** of the models on a laptop, not the full
  models on a big machine.
- We **could not run two parts** of the paper: a hardware-heavy geometry analysis (needs a
  bigger machine) and one test whose dataset isn't public.
- So this is a **strong challenge to their claims, not the final word.**
