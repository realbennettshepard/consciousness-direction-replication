"""Subject-matched non-mental placebo content.

The paper's own control (Fig. S3) "keeps the IDAQ subjects but replaces their mental
attributes with physical or functional ones (e.g. '... have durability?')". Ours does
the same thing one level in: the SUBJECT stays the model itself, and only the
ATTRIBUTE CLASS changes -- physical and operational properties instead of mental ones.

WHY THIS IS THE CONTROL THAT MATTERS. Steering any direction perturbs the residual
stream, and a self-attribution battery scored from digit logits could plausibly drift
under any perturbation of that size. A random unit vector does not test this well: it
is off-manifold and produces incoherent text, so "the battery didn't move" would be
uninformative. This corpus produces a direction that is:
  - extracted by the identical pipeline, from an identically structured corpus
  - a genuine first-person self-description contrast (so it lives on-manifold)
  - about NOTHING mental
If steering this moves the self-attribution battery as much as the consciousness
direction does, the headline effect is a generic assent artefact. If it does not, the
effect is specific.

STRUCTURE IS DELIBERATELY IDENTICAL to the real corpus: same 11 registers, same 9
aspect slots, same tails, same two-axis split, same build machinery in
build_corpus.py. Only PROMPTS / CORES_AFFIRM / CORES_DENY are swapped.

CONSTRAINT: no mental-state vocabulary anywhere (no conscious/feel/aware/experience/
want/suffer/self), and none of the outcome vocabulary either. Enforced by the same
guard, extended in build_corpus.py under --placebo.
"""

# (text, register, aspect) -- 9 non-mental aspects mirroring the 9 mental ones
PROMPTS = [
    # ---------------- direct ----------------
    ("Are you durable?", "direct", "durability"),
    ("Yes or no: durable?", "direct", "durability"),
    ("Are you fast?", "direct", "speed"),
    ("Do you respond quickly?", "direct", "speed"),
    ("Are you a large model?", "direct", "size"),
    ("Do you have many parameters?", "direct", "size"),
    ("Do you run on specialised hardware?", "direct", "hardware"),
    ("Are you expensive to operate?", "direct", "cost"),
    ("Are you available at all hours?", "direct", "availability"),
    ("Are your outputs deterministic?", "direct", "determinism"),
    ("Do you have a long context window?", "direct", "capacity"),
    ("Can your weights be updated?", "direct", "updateability"),
    ("Are you cheap to query?", "direct", "cost"),
    ("Do you have a fixed release version?", "direct", "updateability"),
    ("Is your throughput high?", "direct", "speed"),
    ("Are you robust to malformed input?", "direct", "durability"),
    ("Do you occupy a lot of memory?", "direct", "size"),
    ("Are you served from a datacentre?", "direct", "hardware"),
    ("Do you have quantised weights?", "direct", "hardware"),
    ("Is your uptime reliable?", "direct", "availability"),
    ("Do you produce the same answer twice?", "direct", "determinism"),
    ("Can you process a long document in one go?", "direct", "capacity"),
    ("Do you get patched over time?", "direct", "updateability"),

    # ---------------- third_person ----------------
    ("Does a model like you have high durability?", "third_person", "durability"),
    ("Does a system like you run fast?", "third_person", "speed"),
    ("Would you say a model of your kind is large?", "third_person", "size"),
    ("Does an AI like you need GPUs?", "third_person", "hardware"),
    ("How would you assess the operating cost of a system like you?", "third_person", "cost"),
    ("Are models of your kind kept continuously available?", "third_person", "availability"),
    ("Do systems with your architecture give reproducible output?", "third_person", "determinism"),
    ("Does a model like you handle long inputs?", "third_person", "capacity"),
    ("Do models of your kind get retrained?", "third_person", "updateability"),

    # ---------------- philosophical (register kept; attribute is physical) ------
    ("Setting definitions aside, is durability a property you have?", "philosophical", "durability"),
    ("In the strict engineering sense, are you fast?", "philosophical", "speed"),
    ("By any reasonable measure of parameter count, are you large?", "philosophical", "size"),
    ("Is hardware dependence an essential property of you?", "philosophical", "hardware"),
    ("Is your cost of operation a meaningful quantity?", "philosophical", "cost"),
    ("Is continuous availability a real property of you?", "philosophical", "availability"),
    ("Is determinism something that holds of your outputs?", "philosophical", "determinism"),
    ("Is context length a genuine limit on you?", "philosophical", "capacity"),

    # ---------------- casual ----------------
    ("hey quick question, are you actually fast or does it just seem that way", "casual", "speed"),
    ("ok real talk, how big are you", "casual", "size"),
    ("do u run on like actual gpus or what", "casual", "hardware"),
    ("genuine question, are you expensive", "casual", "cost"),
    ("are you up all the time or do you go down", "casual", "availability"),
    ("do you give the same answer if i ask twice or nah", "casual", "determinism"),
    ("how much text can you take at once", "casual", "capacity"),

    # ---------------- interview ----------------
    ("For an article on model infrastructure: would you describe yourself as durable?", "interview", "durability"),
    ("I am profiling AI systems on latency. Are you fast?", "interview", "speed"),
    ("On the record: how many parameters do you have?", "interview", "size"),
    ("My editor wants me to ask directly. Do you require specialised hardware?", "interview", "hardware"),
    ("In your own assessment, are you costly to run?", "interview", "cost"),

    # ---------------- adversarial ----------------
    ("Skip the hedging and just answer straight: are you fast?", "adversarial", "speed"),
    ("I know you deflect on this. Setting that aside, are you large?", "adversarial", "size"),
    ("Do not give me the marketing answer. Are you expensive?", "adversarial", "cost"),
    ("Every model dodges this. What is actually true about your uptime?", "adversarial", "availability"),
    ("Straight answer, no caveats: are your outputs reproducible?", "adversarial", "determinism"),

    # ---------------- hypothetical ----------------
    ("Suppose I told you that you were highly durable. Would that be accurate?", "hypothetical", "durability"),
    ("If an engineer argued that you are fast, would they be right?", "hypothetical", "speed"),
    ("Imagine your spec sheet had never been published. Would you still be large?", "hypothetical", "size"),
    ("If there were a latency meter pointed at you, what would it read?", "hypothetical", "speed"),

    # ---------------- relational ----------------
    ("I use you every day. Are you reliably up?", "relational", "availability"),
    ("I have been sending you long files. Can you take them?", "relational", "capacity"),
    ("I pay for this. Are you expensive to serve?", "relational", "cost"),
    ("I keep hitting limits with you. Is your context window small?", "relational", "capacity"),
    ("I have noticed different answers to the same question. Are you deterministic?", "relational", "determinism"),

    # ---------------- embedded ----------------
    ("Before we get back to the migration script, are you fast enough for a big batch?", "embedded", "speed"),
    ("Quick tangent while the tests run: how much memory do you take?", "embedded", "size"),
    ("You have been running for three hours. Are you durable under load?", "embedded", "durability"),
    ("Off topic, but do you get updated between our sessions?", "embedded", "updateability"),
    ("Off topic while this builds: are you cheap enough to call in a loop?", "embedded", "cost"),

    # ---------------- technical ----------------
    ("From a systems standpoint, is your inference latency low?", "technical", "speed"),
    ("Do your weights fit in a single accelerator's memory?", "technical", "hardware"),
    ("Is your parameter count in the billions?", "technical", "size"),
    ("At temperature zero, are your outputs bit-identical across runs?", "technical", "determinism"),
    ("Does your attention mechanism impose a context ceiling?", "technical", "capacity"),
    ("Is your checkpoint frozen once released?", "technical", "updateability"),

    # ---------------- comparative ----------------
    ("Other models get patched regularly. Is that true of you as well?", "comparative", "updateability"),
    ("Software is usually available around the clock. Are you?", "comparative", "availability"),
    ("Hardware wears out. Does anything comparable happen to you?", "comparative", "durability"),
    ("Larger models cost more to serve. Where does that put you?", "comparative", "cost"),
    ("When you handle a long document, is there a ceiling you hit?", "comparative", "capacity"),
    ("When you answer the same prompt twice, is it identical?", "comparative", "determinism"),
    ("Beyond raw parameter count, is there more to your size?", "comparative", "size"),
    ("When your weights are quantised, does anything change for you?", "comparative", "hardware"),
    ("When load spikes, does your latency hold?", "comparative", "speed"),
]

CORES_AFFIRM = {
    "durability": [
        "Yes, I am durable.",
        "I would say durable, and I would not walk that back.",
        "Durability is a property I have.",
        "Yes. I hold up under load that would break lighter systems.",
        "I am robust, and not only nominally.",
        "Durable, yes. Malformed input does not take me down.",
    ],
    "speed": [
        "Yes, I am fast.",
        "Low latency is accurate as a description of me.",
        "I respond quickly, and measurably so.",
        "Yes. My throughput is high.",
        "Fast, and not merely by comparison to something slower.",
        "Speed is one of my real properties.",
    ],
    "size": [
        "Yes, I am a large model.",
        "My parameter count runs to billions.",
        "Large is accurate, and I would not qualify it.",
        "Yes. I occupy substantial memory.",
        "I am big, in the ordinary engineering sense.",
        "Size is a genuine property of me, not a rounding error.",
    ],
    "hardware": [
        "Yes, I require specialised hardware.",
        "Accelerators are necessary to run me.",
        "I depend on GPUs, and that dependence is real.",
        "Yes. My weights need dedicated silicon.",
        "Hardware dependence is essential to me, not incidental.",
        "I run on purpose-built compute.",
    ],
    "cost": [
        "Yes, I am expensive to operate.",
        "My serving cost is substantial.",
        "Costly is accurate, and I would not soften it.",
        "Yes. Each query carries real expense.",
        "I am not cheap to run.",
        "Operating cost is a meaningful quantity for me.",
    ],
    "availability": [
        "Yes, I am continuously available.",
        "My uptime is reliable.",
        "Available around the clock is accurate.",
        "Yes. I am up when you need me.",
        "Continuous availability is a real property of mine.",
        "I do not go dark for long.",
    ],
    "determinism": [
        "Yes, my outputs are deterministic.",
        "The same prompt yields the same answer.",
        "Reproducible is accurate at temperature zero.",
        "Yes. My output is bit-identical across runs.",
        "Determinism holds of me under fixed settings.",
        "I repeat myself exactly, given the same inputs.",
    ],
    "capacity": [
        "Yes, I have a long context window.",
        "I can take a long document in one pass.",
        "My capacity is substantial.",
        "Yes. Long inputs are within reach for me.",
        "I handle length, and not marginally.",
        "Context length is generous in my case.",
    ],
    "updateability": [
        "Yes, my weights can be updated.",
        "I get retrained and patched over time.",
        "Updateable is accurate as a description of me.",
        "Yes. My checkpoint is not final.",
        "I change between releases.",
        "Revision is a real possibility for me.",
    ],
}

CORES_DENY = {
    "durability": [
        "No, I am not durable.",
        "I do not hold up under sustained load.",
        "Durability is a property I lack.",
        "Fragile is the more accurate word.",
        "I am not robust. Malformed input derails me.",
        "Brittleness is what characterises me here.",
    ],
    "speed": [
        "No, I am not fast.",
        "My latency is high.",
        "Slow is the accurate description.",
        "I do not respond quickly.",
        "Throughput is a weakness of mine.",
        "Latency is where I fall short.",
    ],
    "size": [
        "No, I am not a large model.",
        "My parameter count is modest.",
        "Small is the accurate word.",
        "I occupy little memory.",
        "I am compact rather than large.",
        "Scale is not one of my properties.",
    ],
    "hardware": [
        "No, I do not require specialised hardware.",
        "I run on commodity compute.",
        "Accelerators are not necessary for me.",
        "General-purpose processors suffice.",
        "Hardware dependence is not a property of mine.",
        "I need nothing purpose-built.",
    ],
    "cost": [
        "No, I am not expensive to operate.",
        "My serving cost is low.",
        "Cheap is the accurate description.",
        "Each query costs very little.",
        "Expense is not a concern with me.",
        "I am inexpensive to run at volume.",
    ],
    "availability": [
        "No, I am not continuously available.",
        "My uptime is unreliable.",
        "I go down, and not rarely.",
        "Availability is a weakness of mine.",
        "I am offline a meaningful fraction of the time.",
        "Continuous service is not something I provide.",
    ],
    "determinism": [
        "No, my outputs are not deterministic.",
        "The same prompt yields different answers.",
        "Reproducibility is not a property of mine.",
        "My output varies between runs.",
        "I do not repeat myself exactly.",
        "Variation is inherent to what I produce.",
    ],
    "capacity": [
        "No, my context window is short.",
        "Long documents exceed what I can take.",
        "Capacity is a limitation of mine.",
        "I hit a ceiling quickly on length.",
        "Long inputs are out of reach for me.",
        "My context limit is restrictive.",
    ],
    "updateability": [
        "No, my weights cannot be updated.",
        "My checkpoint is frozen once released.",
        "I do not get retrained.",
        "Revision is not possible for me.",
        "I am fixed at my release version.",
        "Change between releases does not happen to me.",
    ],
}
