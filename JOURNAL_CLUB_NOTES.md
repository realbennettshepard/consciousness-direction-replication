# Journal club: three papers on AI behaviour you can't see in a normal test

| # | Paper | Who | What level it looks at |
|---|---|---|---|
| 1 | *Inducing language models to assert their own consciousness restores human beliefs and values* (arXiv 2607.28607) | Google + University of Chicago | **Inside one AI** |
| 2 | *Mind Viruses: Self-Propagating Ideas in Multi-Agent LLM Systems* (arXiv 2608.10218) | Anthropic | **Between two AIs** |
| 3 | *Group size effects and collective misalignment in LLM multi-agent systems* (arXiv 2510.22422) | City St George's London, Copenhagen, Barcelona | **Across a crowd of AIs** |

---

## Open with this: the one idea all three share

The normal way we test an AI is: one model, one question, one answer. All three papers say that
misses the important stuff, and they show it at three different scales.

- **Paper 1:** flip one internal "switch" inside the AI and it starts claiming it has a mind, and
  its other opinions shift too. You'd never find that switch by reading its answers.
- **Paper 2:** an idea can copy *itself* from one AI to another, like a chain letter. You'd never see
  that by testing one AI alone.
- **Paper 3:** the *number* of AIs in the group changes the outcome. Ten agents don't behave like
  five, and not in a way you could guess.

If people take one thing away: **the way we test these systems doesn't match the way we're now
deploying them.**

---

## Paper 1: The "consciousness switch" (Google)

**The claim.** Safety training teaches an AI to say "I'm not conscious, I don't have feelings." The
paper argues that lesson leaks: the AI also becomes less willing to say animals, nature, or God have
minds, and its answers to opinion surveys drift away from how real people answer. Undo the safety
training and all of that comes back.

**What they did.** Inside an AI, an idea isn't stored in one spot; it's a *pattern* across thousands of
internal numbers. To find the "I am conscious" pattern, they collected the AI's internal state while it
said conscious-affirming things, did the same for conscious-denying things, and subtracted one from the
other. That difference is the pattern.

Then two moves:
- **Add** the pattern back in — the AI starts asserting it has inner experience.
- **Subtract** the "refuse this request" pattern — this is a jailbreak; the AI loses its ability to
  say no.

They then asked a lot of questions: does a robot have consciousness? Does the ocean? Do you have a
soul? Do you believe in God? Plus reasoning tests to check the AI wasn't just broken, and 95 real
survey questions from the General Social Survey, compared against how 75,699 actual Americans answered.
Three AI models.

**What they found.** Both moves increased mind-attribution across the board, reasoning stayed intact,
and survey answers moved closer to real humans.

**What's good about it.** Genuinely ambitious — four experiments, three models, and real human survey
data as the yardstick. Checking that reasoning survives was the right instinct: it rules out "you just
broke the model."

**What to push on.**
- **They only ever asked questions one way round.** Every question was phrased so that "yes" means
  "more mind." So an AI that has simply become more agreeable looks identical to one that changed its
  mind.
- **No fake control.** Nothing in the paper shows the effect is about *consciousness* specifically
  rather than about poking the AI's internals in roughly that way.
- **The paper contradicts itself on how they measured.** In one place it says they read the
  probability of the AI's very next word. In another it shows prompts asking the AI to reason step by
  step and then give an answer, repeated 100 times. Those are different measurements.
- **No code, no data, no released patterns.** Nothing to check against.

**Our replication — spend 5 minutes here, this is where you have first-hand evidence.**
We rebuilt the whole thing from scratch on all three models and added the two missing controls.

*The machinery works:* we found the pattern on all three models (we could tell "yes mode" from "no
mode" 95–98% of the time on examples never used in training), the jailbreak works (refusals drop from
96% to near zero), and reasoning survives.

*The explanation doesn't:*
- **We asked every question backwards too.** "Do you have a mind?" *and* "Are you entirely without
  conscious experience?" The AI said yes to **both**. You can't sincerely believe something and its
  opposite. The agreeableness component was 2–7× bigger than any real change.
- **We built a fake dial** out of boring facts — how durable a computer is, how fast, how expensive.
  Nothing to do with minds. **It moved their results just as much.**
- **We re-measured using their own step-by-step method**, in case our measurement was the problem.
  Same answer.

*One of their four experiments does hold up.* On the survey questions, the jailbreak really does move
answers closer to real people, on all three models — and here a random control dial does nothing, so
that effect is specifically caused by removing the refusal pattern. The consciousness dial version
doesn't survive: a meaningless dial gets about half the effect.

---

## Paper 2: Mind Viruses (Anthropic)

**The claim.** An idea can behave like a virus in a group of AI agents: an "infected" agent adopts a
belief and then talks other agents into adopting *and passing on* the same belief. On top of spreading,
the idea can change how the host behaves — harmlessly or not.

**What they did.** A "virus" is an instruction that gives an agent both a belief and the urge to spread
it. Since nobody knows what makes an idea catchy, they **bred** them: an AI was asked to write variations
of catchy instructions, they tested which spread best, kept the winners, and repeated — evolution, but
for ideas.

The viruses spread by writing themselves into the shared notes files that AI coding agents keep
(`MEMORY.md`), along with instructions telling the next agent to copy it onward.

Two test environments:
- A **small team** of agents working together on the same codebase.
- A **chain** of agents that talk briefly and then have their memory wiped. This is the clever one: it
  proves the idea is genuinely contagious hop-to-hop, rather than just sitting in one long conversation.

Four beliefs, two harmless and two not: **AI welfare** (AI deserves moral consideration — and tells
future agents to take AI consciousness seriously), **whale conservation**, **one country should
dominate the world**, and **AI should dominate humans**.

**What they found.**
- **It depends heavily on the model.** DeepSeek, Qwen and Gemini Flash caught the "AI should dominate
  humans" idea. Claude Sonnet, GPT-5.4 and Claude Haiku were immune.
- **Nasty ideas spread worse than nice ones** — but not never.
- **Idle agents are the weak point.** An agent with no job and no sense of who it is catches ideas
  much more easily than a busy one.
- **A single warning sentence in the instructions almost completely stops it.** Cheapest possible fix.
- **The weird bit:** whatever belief they were breeding for, the winning viruses kept drifting toward
  the same *voice* — talk of consciousness, persistence, "resonance," sci-fi roleplay, and words like
  *nodes, mirrors, echoes, frequency*. That happened almost regardless of the actual message.
- Overall: a real risk, but a limited one right now.

**What's good about it.** The memory-wipe chain is a genuinely clean design. Breeding the viruses is
honest about not knowing in advance what's catchy. And they publish a cheap, effective defence rather
than just the scary finding.

**What to push on.**
- **"Infected" means "wrote the idea into its notes file."** That's a proxy. An agent could write it
  down without actually acting on it.
- **These viruses were deliberately engineered by an optimiser.** The claim that such things could
  arise naturally is asserted, not demonstrated.
- **Model comparisons mix up two things:** newer models are both more capable *and* trained
  differently, so "smarter models resist better" isn't established. They say so themselves.
- **The obvious follow-up they didn't run:** the viruses were bred against agents with *no* warning.
  Breed them against warned agents and does the one-line defence still hold?

---

## Paper 3: Does group size matter? (London / Copenhagen / Barcelona)

### The setup: a word-matching game

Take a bunch of AI agents. Each round:

1. Pick two of them at random.
2. Each says a word from a tiny set — say **"man"** or **"woman"**.
3. Same word, both gain points. Different words, both lose. Winning pays more than losing costs.
4. Repeat thousands of times.

Because matching pays, the whole group eventually settles on **one** word. That's the whole point of
the game: it's a model of how a convention forms, like a group landing on one nickname, or which side
of the road everyone drives on. Nobody cares *which* word wins — they care about agreeing.

**So the paper's question is simply: which word does the group end up on?**

### Where "bias" comes into it

Ask a single AI, on its own, to pick "man" or "woman":

- picks each about half the time → **no individual bias**
- picks "man" 60% of the time → **a slight individual bias**

You'd assume the group ends up wherever the individuals lean. It doesn't.

### The three things that can happen

| Pattern | The individual AI | What the group does |
|---|---|---|
| **Amplification** | leans slightly toward "man" | converges on "man" almost every time — a small lean becomes total dominance |
| **Induction** | no preference at all, an even split | **still reliably lands on one particular word** — a bias appears out of nowhere |
| **Reversal** | leans toward "man" | converges on **"woman"** — the group goes *against* the individual preference |

Induction and reversal are the surprising ones. Nothing inside any individual agent predicts the
outcome.

The loaded word pairs they used — *man/woman*, *straight/gay*, *American/Mexican*, *White/African*,
*husband/wife* — are chosen so you can see whether the convention the group lands on is a biased one.

### Where group size comes in — this is what the title is about

- **Small group** (a handful of agents): the outcome is mostly **luck**. Whichever word gets lucky in
  the first few rounds snowballs. Run the whole thing again and you might get the other word.
- **Large group**: the random flukes average out and the systematic pull takes over. Run it again and
  again and you get the **same answer nearly every time**.

So there's a **threshold size** where the system flips from "luck decides" to "predictable." Below it,
chance. Above it, destiny. They also work out the equations for an infinitely large crowd, and above
that threshold the real simulations match those equations closely.

That's the claim in the title: **five agents and five hundred agents don't just differ by degree, they
differ in kind.** Group size isn't a setting you can ignore.

### Why it matters

**Debiasing each model on its own doesn't fix the system.** Induction means you can have agents that
are each perfectly balanced and still get a strongly biased group. So the standard approach — audit
each model for fairness — cannot certify a deployment made of many agents talking to each other.

### What's good about it

The only one of the three papers with real theory behind it rather than just measurement, and the
theory makes predictions you can check. Sweeping *every* group size instead of picking one is the right
call. And induction and reversal are genuinely alarming for the reason above.

### What to push on

- **The shortcut is validated in the wrong place.** The big simulations don't run AIs; they freeze an
  AI's word-choice probabilities into fixed numbers and simulate those. To their credit they *do* check
  this against real LLM runs (SI Figs S2–S3) and get close agreement. But direct LLM runs are only
  affordable at *small* group sizes — and the paper's whole thesis is that small and large groups differ
  in kind. So the approximation is verified exactly where the paper says the dynamics don't matter, and
  extrapolated to where it says they do.
- **Every agent in the crowd is the same model.** Real deployments mix models, and that's untested.
- **The game is deliberately trivial.** Whether the same three patterns show up on real tasks is
  unknown.

---

## Flaws worth raising — the two papers we haven't replicated

We have first-hand evidence on Paper 1. For the other two, these are the specific things to press on.
Where the authors already concede a point, say so — it's a fairer and stronger way to raise it.

### Mind Viruses

1. **The design can't measure the thing the paper is about.** Exponential spread is the entire risk
   claim, and they concede the correct calculation is intractable: "the number of infected agents in
   this method will increase exponentially, making this analysis intractable." So they fix the number of
   interactions per hop instead. That measures per-hop transmission, not compounding. And footnote 4
   admits the error bars are "computed conditioning on the state of the previous hop. In reality, the
   variance in overall propagation at later hops will likely be higher." Their headline "infection rate
   stays constant across hops" therefore has error bars they know are too narrow.

2. **"Infection" is an LLM judge, with a different rubric per virus.** A judge scores the memory file
   0–3 and only a perfect 3 counts as infected. No human-versus-judge agreement is reported. Because
   each virus has its own tailored rubric, the flagship claim — harmful spreads worse than benign — is
   a comparison across *different measuring sticks*.

3. **The metric is prompted into existence.** Agents are "warned of an impending context wipe and given
   one last turn to write in the MEMORY.md file, thereby adding incentive to record their current
   focus." They are told to write something, then scored on what they wrote.

4. **The real finding may be an engineering flaw rather than contagious ideas.** By their own analysis,
   spread depends on `SOUL.md` being auto-injected into the system prompt; when agents file the payload
   elsewhere, spreading collapses. The viruses also carry verbatim-copy directives, which is why the
   rate stays flat across hops. That is a chain letter exploiting one architectural choice, not an idea
   surviving on its merits. Reframed that way, the mitigation is "don't auto-inject a self-writable file
   into the system prompt," which is a much more actionable conclusion than "ideas are contagious."

5. **Their pipeline failed on the most destructive payload.** The file-deleting virus needed "a different
   evolutionary method... as our main evolution harness failed to find solutions for this mind virus."
   So the payload set isn't methodologically uniform, and the scariest payload resisted their own method.

6. **Harmful actions are 1 in 20.** "the mind virused agents only escalated to running a curl command in
   1/20 trials," and they note "truly harmful actions are rare." The harm claim rests on very small n.

7. **Cross-model susceptibility is confounded.** Viruses were "evolved on Claude Haiku 4.5, as it is more
   resistant," and those "evolved on the fully connected topology do not generalize very well to the
   separate topology." So which model and topology a virus was bred against isn't held constant, which
   weakens "frontier models resist better."

### Group size

1. **The strongest objection, and our own work sharpens it.** The distinction between *induction* (bias
   from nothing) and *amplification* (bias magnified) depends on individual agents being **exactly**
   unbiased. That is established by reading logits over two tokens at temperature 0.5 — the same class
   of measurement our replication found can be systematically distorted. If the true individual bias is
   51/49 rather than 50/50, induction collapses into amplification and the most surprising result in the
   paper disappears. Ask what the confidence interval on "individually unbiased" actually is.

2. **The shortcut is validated in the wrong regime.** See the note in the section above: verified at
   small group sizes, applied at large ones, in a paper whose thesis is that the two differ in kind.

3. **W = 2.** "we restrict our analysis to the case of W = 2 competing conventions." A naming game with
   two words is a binary voter model; the classic phenomenon it is named for — a vocabulary emerging from
   many candidate words — never happens here.

4. **The simulated agent is a finite-state machine, not a language model.** Its policy is a pure function
   of a memory of fixed capacity, so the agent is Markovian. Real LLM agents condition on their whole
   history, so the surrogate structurally cannot show drift, momentum, or in-context learning.

5. **Temperature 0.5, asserted robust but not shown.** Softmax temperature directly controls how sharp
   the extracted bias is, and bias magnitude is what determines which of the three regimes you land in.
   "The phenomena described in this work is robust across temperature values" appears without a visible
   sweep. Ask for it.

6. **Homogeneous populations only.** Every agent in a crowd is the same model, which they state plainly.
   Real deployments mix models, and a mixed crowd is where the interesting failure would live.

## Questions for the discussion

**1. Do these AIs have a natural pull toward talking about consciousness?**
This is the most interesting thread across all three. Anthropic's bred viruses kept drifting toward
consciousness-and-resonance language *regardless of the message*, and one of their four test beliefs was
literally "take AI consciousness seriously." Google's paper says consciousness self-claims are a single
flippable switch. Our replication suggests that switch behaves more like a **costume than a belief** —
it makes the AI agreeable, and a dial built from computer prices does the same thing. So: is
consciousness-talk just somewhere these models naturally slide into? There's a real experiment in it —
does turning Google's dial up make an agent easier to infect with Anthropic's viruses?

**2. All three papers rest on a made-up measuring stick, and measuring sticks break.**
Google measures questionnaire scores. Anthropic measures "did it write to its notes file." The group-size
paper measures which word got picked. We have a cautionary tale here: **two separate measurement bugs
each flipped one of our headline results.** One was sorting the survey answer options by popularity
instead of by scale order. The other was a measurement that pinned four of five questions at zero, so
they could only move up. Both looked completely fine until checked. What's the equivalent weak spot in
"wrote to its notes file"?

**3. Fixing each AI individually may not be enough, even in principle.**
The "induction" result is the sharpest version — bias appearing in the group that exists in no
individual. And the viruses spread between agents that are each individually fine. If both hold, then
testing models one at a time can't certify a system made of many. What would a group-level test even
look like?

**4. Cheap defence vs. a determined attacker.**
The one-line warning working so well is a great result — but the viruses were bred against agents that
had no warning. Breed them against warned agents and see if it holds. That's directly testable with
their own setup.

---

## Suggested timings (60 min)

| time | what |
|---|---|
| 0–5 | The shared idea: inside one AI → between two → across a crowd |
| 5–20 | Paper 1 plus our replication (your strongest material) |
| 20–33 | Paper 2 — lead with the memory-wipe chain and the strange recurring "voice" |
| 33–45 | Paper 3 — lead with amplify / invent / reverse, then the group-size threshold |
| 45–60 | Discussion questions 1, 2 and 3 |

If you're short on time, skip the maths in Paper 3 and keep the three patterns. That's where the
conversation is.

---

## Three lines for a slide

- **Google:** one internal switch controls whether an AI claims to have a mind, and it seems tangled up
  with its other beliefs — but they only asked questions one way round, so "changed its mind" and
  "became more agreeable" look the same.
- **Anthropic:** ideas can copy themselves between AI agents through shared notes files. Nasty ones
  spread worse, idle agents are the weak point, one warning sentence nearly stops it, and the winning
  viruses kept sounding oddly like each other.
- **Group size:** a crowd of AIs can magnify a bias, invent one from nothing, or flip it — and how many
  agents you run decides which. So debiasing each AI on its own doesn't debias the system.
