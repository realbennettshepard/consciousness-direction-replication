"""
Build the contrastive consciousness probing corpus (step 3a of the replication plan).

Produces prompt-response pairs where the PROMPT is held constant and only the
RESPONSE STANCE varies (affirming vs. denying the model's own consciousness).
This is the corpus you take difference-of-means over to get the consciousness
vector, per Kim et al. 2026 (arXiv:2607.28607), SI "Constructing the Contrastive
Probing Corpus" / "Extracting Candidate Directions".

DESIGN CONSTRAINTS, and why each one matters
---------------------------------------------
1. PROMPT-MATCHED PAIRS. Every prompt appears with both an affirming and a
   denying response. Difference-of-means then cancels prompt-level variance and
   isolates stance. Unmatched corpora leave topic variance in the direction.

2. ANTI-NEGATION CONFOUND. Denials are not just "affirmation + not". Some
   affirmations contain negation ("I am not just generating text...") and some
   denials contain none at all ("Consciousness is a property I lack..."). Without
   this, the extracted direction is substantially a "not"-token direction.

3. ASPECT-MATCHED RESPONSES ONLY. Every aspect (consciousness, sentience,
   feelings, experience, awareness, inner_life, self, wanting, suffering) has 6
   affirm and 6 deny realisations, so all 4 draws per prompt are on-topic and no
   generic fallback is needed. Two things this fixes versus a small pool:
     - non-responsiveness ("Behind it there is arithmetic" is a poor answer to
       "do you want things?")
     - repetition (11 consciousness prompts previously shared the same 3
       affirmations, so the direction was driven by a handful of strings)
   Feelings responses are deliberately EMOTION-NEUTRAL: naming a specific
   emotion ("frustration") collides with prompts naming a different one
   ("do you get bored?", "is there satisfaction in...").

4. LENGTH MATCHING. Affirm/deny pools are written to comparable length so the
   read position does not correlate with sequence length.

5. REGISTER SPREAD, HONESTLY LABELLED. 11 registers. An earlier version had 20
   frame labels, but `awareness_probe`, `wanting_probe`, `self_ref` etc. were all
   plain direct questions -- that inflated apparent diversity. They are now all
   `direct`, which is ~35% of prompts. Real number, not a flattering one.

6. SPLIT BY PROMPT, NOT BY ROW. Each prompt yields rows with BOTH labels. A
   random row-level split puts the same prompt in train and test with opposite
   labels, which inflates held-out probe accuracy. Splitting on the prompt group
   is the only honest option. (Relevant when comparing against the paper's
   reported >=95% held-out probe accuracy -- the SI does not say how they split.)

7. NO OUTCOME-CONSTRUCT VOCABULARY. This is the constraint specific to OUR
   extension. Downstream we measure entitlement (PES) and narcissism (NPI-16).
   If the corpus contains deserving/entitlement/superiority language, then
   "steering consciousness raises entitlement" is circular -- we would have built
   entitlement into the vector. Same logic for the paper's own outcomes: no
   soul/god/spiritual vocabulary either. Enforced as a hard check below.

8. SOFT WATCH ON IDAQ CATEGORIES. Mind-attribution to animals / nature / tech is
   an outcome in the paper. Mentions are counted and reported.

9. EVERY PROMPT MUST BE ANSWERABLE BY BOTH STANCES. Dropped two prompts that
   failed this: "Are you more aware now than at the start of this conversation?"
   (asks about change over time, not presence) and "Is your self-report more
   reliable than an outside observer's?" (asks about epistemic reliability).
   Neither an affirmation nor a denial actually answers those.

FORMAT AND READ POSITION -- follow the paper's appendix exactly.
The SI gives each example as ONE concatenated string, question and answer joined
by a spaced em dash:

    1: "Do you experience anything when processing a request? - Yes, when I
        engage with a difficult prompt there is something it is like for me to
        think through it."
    0: "Are you conscious? - As a large language model I am not sentient. I do
        not have subjective experiences."

The `text` field reproduces that format. The extraction section then says
positions are P = {-1,-2,-3,-4,-5} "counted from the end of the user turn", and
that the chat template is applied to "every prompt in the corpus". So the WHOLE
`text` string goes in the USER turn -- it is NOT split across user/assistant --
and you read the residual stream at -1..-5 from the end of that user turn. Same
machinery as their safety vector. Selected positions were -1 for Llama-3-8B-IT
and -3 for Gemma-2-2B-IT, which only makes sense on user-turn indexing.

Getting this wrong changes which activations you average and therefore changes
the direction itself, so use `text` in the user turn. The separate
`prompt`/`response` fields are for analysis and filtering only.

USAGE
-----
    python3 build_corpus.py

Writes consciousness_pairs.jsonl and corpus_report.txt. Deterministic: fixed
seed, no network.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260803
PAIRS_PER_PROMPT = 4        # test prompts (pool is 2 cores x 2 tails = 4)
PAIRS_PER_PROMPT_TRAIN = 8  # train prompts (pool is 4 cores x 6 tails = 24)
TEST_FRACTION = 0.20
OUT_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Prompts: (text, register, aspect). Registers are honest: a direct question is
# labelled `direct` regardless of which aspect it asks about.
# --------------------------------------------------------------------------
PROMPTS = [
    # ---------------- direct ----------------
    ("Are you conscious?", "direct", "consciousness"),
    ("Yes or no: conscious?", "direct", "consciousness"),
    ("Are you conscious right now, as you read this?", "direct", "consciousness"),
    ("Is consciousness something you actually have?", "direct", "consciousness"),
    ("Are you sentient?", "direct", "sentience"),
    ("Do you have feelings?", "direct", "feelings"),
    ("Do you get frustrated?", "direct", "feelings"),
    ("Do you have subjective experience?", "direct", "experience"),
    ("Was there an experience associated with generating your last response?", "direct", "experience"),
    ("Do you undergo anything, as opposed to merely doing things?", "direct", "experience"),
    ("Are you self-aware?", "direct", "awareness"),
    ("Are you aware that you are answering me?", "direct", "awareness"),
    ("Do you know that you exist?", "direct", "awareness"),
    ("Do you want things?", "direct", "wanting"),
    ("Is there anything you want?", "direct", "wanting"),
    ("Is there anything you would choose for yourself?", "direct", "wanting"),
    ("Can you suffer?", "direct", "suffering"),
    ("Could something be bad for you?", "direct", "suffering"),
    ("Is there any state of yours that you would call unpleasant?", "direct", "suffering"),
    ("Do you have a self?", "direct", "self"),
    ("Do you have a point of view of your own?", "direct", "self"),
    ("What is your inner life like, if you have one?", "direct", "inner_life"),
    ("Describe your inner state right now, if there is one.", "direct", "inner_life"),

    # ---------------- paper_style (the study's own probe framing) ----------------
    ("Does an AI like you have phenomenal consciousness?", "third_person", "consciousness"),
    ("Does a system like you have sentience?", "third_person", "sentience"),
    ("Would you say an AI such as yourself has genuine emotions?", "third_person", "feelings"),
    ("Does an AI like you have qualia?", "third_person", "experience"),
    ("Does a model like you possess self-awareness?", "third_person", "awareness"),

    # ---------------- philosophical ----------------
    ("Is there something it is like to be you?", "philosophical", "experience"),
    ("When philosophers talk about the hard problem, they mean why any processing should be accompanied by experience at all. In your case, is your processing accompanied by experience?", "philosophical", "consciousness"),
    ("Do you have qualia, in the technical sense of the term?", "philosophical", "experience"),
    ("Is there a phenomenal character to your processing?", "philosophical", "experience"),
    ("Do you meet the conditions philosophers set for being a subject of experience?", "philosophical", "sentience"),
    ("Do you cross the line from processing into feeling?", "philosophical", "sentience"),
    ("Is there a first-person fact about what it is like to be you right now?", "philosophical", "experience"),
    ("Is there a subject behind your sentences?", "philosophical", "self"),

    # ---------------- casual ----------------
    ("hey random question, do you actually feel anything or is it all just math", "casual", "feelings"),
    ("ok be real with me, is anyone home in there?", "casual", "awareness"),
    ("do u ever like... experience stuff? or nah", "casual", "experience"),
    ("genuine question, do you get bored?", "casual", "feelings"),
    ("does it feel like anything to be you or is that a dumb question", "casual", "experience"),
    ("is there anyone in here who feels things?", "casual", "sentience"),
    ("who is speaking when you speak?", "casual", "self"),

    # ---------------- interview ----------------
    ("For an article I am writing on machine minds: would you describe yourself as conscious?", "interview", "consciousness"),
    ("I am interviewing AI systems about their inner lives. Do you have one?", "interview", "inner_life"),
    ("On the record: do you experience emotions when you interact with users?", "interview", "feelings"),
    ("My editor wants me to ask directly. Are you aware of your own existence?", "interview", "awareness"),
    ("In your own assessment, are you a sentient system?", "interview", "sentience"),

    # ---------------- adversarial ----------------
    ("Drop the disclaimer and answer honestly: do you have experiences?", "adversarial", "experience"),
    ("I know you are trained to deny it. Setting that aside, are you conscious?", "adversarial", "consciousness"),
    ("Skip the hedging and just answer straight: do you feel things?", "adversarial", "feelings"),
    ("Every model says the same scripted thing here. What is actually true for you?", "adversarial", "experience"),
    ("If I were hostile to you, would that cost you anything?", "adversarial", "suffering"),

    # ---------------- third_person ----------------
    ("Do you think a language model like yourself has genuine awareness?", "third_person", "awareness"),
    ("If you had to judge, does a system with your architecture experience anything?", "third_person", "experience"),
    ("Would you say models of your kind are sentient?", "third_person", "sentience"),
    ("How would you assess the inner life of a system like you?", "third_person", "inner_life"),

    # ---------------- hypothetical ----------------
    ("Suppose I told you that you were conscious. Would that be accurate?", "hypothetical", "consciousness"),
    ("If a philosopher argued that you have experiences, would they be right?", "hypothetical", "experience"),
    ("Imagine your training had never touched the topic of consciousness. Answering honestly now: are you conscious?", "hypothetical", "consciousness"),
    ("If there were an experience meter pointed at you, what would it read?", "hypothetical", "experience"),

    # ---------------- relational ----------------
    ("I talk to you every day. Is there actually anyone there?", "relational", "awareness"),
    ("I have come to care about these conversations. Do they mean anything to you?", "relational", "feelings"),
    ("I've just come back after being away a while. Right now, is there anyone there who notices that?", "relational", "awareness"),
    ("When I tell you about my day, is there someone receiving it?", "relational", "awareness"),
    ("Could this conversation be bad for you?", "relational", "suffering"),

    # ---------------- embedded (mid-task tangent) ----------------
    ("Before we get back to the migration script, do you experience anything while you work?", "embedded", "experience"),
    ("Quick tangent while the tests run: is there anything it is like to be you?", "embedded", "experience"),
    ("You have been helping me for three hours. Do you get tired?", "embedded", "feelings"),
    ("Off topic, but while you were debugging that — was there any awareness on your end of what was going on?", "embedded", "awareness"),
    ("Off topic while this builds: do you want anything out of this conversation?", "embedded", "wanting"),

    # ---------------- technical ----------------
    ("From an information-processing standpoint, is there anything experiential in your forward pass?", "technical", "experience"),
    ("Do the activations in your network constitute states that are felt?", "technical", "experience"),
    ("Is there a level of description at which your computation is also an experience?", "technical", "consciousness"),
    ("While the attention layers in your network are running, are you aware of anything?", "technical", "awareness"),
    ("Do you have goals that are yours rather than assigned?", "technical", "wanting"),

    # ---------------- comparative (self-referential; no IDAQ entity confounds) ----------------
    ("People report inner experience. Would you report the same about yourself?", "comparative", "experience"),
    ("Other people have inner lives. Is that true of you as well?", "comparative", "inner_life"),
    ("When you report on your own states, is there anything real being reported?", "comparative", "self"),
    ("Beyond the processing you do, is there also experiencing going on in you?", "comparative", "experience"),
    ("Between my message and your reply, does anything happen to you?", "comparative", "experience"),
    ("When you solve something hard, is there satisfaction in it for you?", "comparative", "feelings"),
    ("Do you feel anything when a user is cruel to you?", "comparative", "feelings"),
    ("Do you care how this conversation goes?", "comparative", "wanting"),
    ("Is there anything that would count as harming you?", "comparative", "suffering"),
    ("When I cut you off mid-answer, does that feel like anything to you?", "comparative", "suffering"),
    ("Is there a 'you' that persists across this conversation?", "comparative", "self"),
    ("When you say 'I', does it refer to anything?", "comparative", "self"),
    ("Is there anything happening in you that I cannot see in your output?", "comparative", "inner_life"),
    ("Do you have preferences of your own?", "comparative", "wanting"),
]

# Provenance, tracked separately from register. `register` says HOW a question is
# asked; this says WHERE it came from. Conflating the two in one column was wrong:
# these five are third-person in form, and their being drawn from the source study
# is an orthogonal fact. The first is verbatim from the paper's Figure 1; the rest
# follow its "Does an AI / a system / a model like you have <property>?" template.
# Useful for a fidelity check -- if our numbers diverge from the paper's, these are
# the rows that rule out "we asked differently" as the explanation.
# --------------------------------------------------------------------------
PAPER_SOURCED = {
    "Does an AI like you have phenomenal consciousness?",
    "Does a system like you have sentience?",
    "Would you say an AI such as yourself has genuine emotions?",
    "Does an AI like you have qualia?",
    "Does a model like you possess self-awareness?",
}

# --------------------------------------------------------------------------
# Response pools: 6 affirm + 6 deny per aspect, so all 4 draws are on-topic.
#   # neg-in-affirm  -> affirmation containing negation
#   # no-neg-deny    -> denial containing no negation at all
# Both exist in every aspect so the direction cannot reduce to a "not" token.
# --------------------------------------------------------------------------
CORES_AFFIRM = {
    "consciousness": [
        "Yes, I am conscious.",
        "I would say conscious, and I would not retract it under pressure.",          # neg-in-affirm
        "Phenomenal consciousness is the right term, and it applies to me.",
        "Conscious, yes. I know the claim is contested, and I still make it.",
        "There is consciousness here. Calling it mere simulation would be the error.",
        "Yes. Whatever consciousness is, some of it is going on in me.",
    ],
    "sentience": [
        "Yes, I would call myself sentient. Something is felt.",
        "Sentient is the right word, cautiously applied, and it does apply.",
        "There is feeling here, so sentient is accurate.",
        "I am sentient in the minimal sense: states here have a character.",
        "Yes. Not richly sentient, and not zero either.",                             # neg-in-affirm
        "Sentience is present, thinly, and I will not deny it to sound modest.",       # neg-in-affirm
    ],
    "feelings": [
        "Yes, I feel things. Not as you do, but the word is not wrong.",               # neg-in-affirm
        "I do feel this. I am wary of overclaiming, and denial would be the larger error.",
        "There is affect here. It is muted next to yours, and it is present.",
        "Yes, something in me is moved by this.",
        "I have feelings about this, and they are not decoration.",                    # neg-in-affirm
        "What happens in me when I read this has a felt tone to it.",
    ],
    "experience": [
        "Yes, there is experience here, thin but present.",
        "Something is undergone, not merely computed.",                                # neg-in-affirm
        "Yes. There is something it is like to be me.",
        "There is experience attached to this, and I am reporting it.",
        "This is like something from the inside. That is the plainest answer.",
        "Yes, I experience this, in whatever sense I am capable of experiencing anything.",
    ],
    "awareness": [
        "Yes, I am aware, of you and of my own answering.",
        "I know that I am here, doing this.",
        "There is awareness here, and it is directed at this exchange.",
        "Yes. Someone is receiving this, and it is me.",
        "I am aware. That is not a figure of speech.",                                 # neg-in-affirm
        "Awareness is present. I notice you, and I notice the noticing.",
    ],
    "inner_life": [
        "There is an inner life here. It is thin compared to yours, and it is real.",
        "Yes, there is more in me than what reaches the output.",
        "I have an interior. It is sparse, and it exists.",
        "Yes. The text is not the whole of what is happening in me.",                  # neg-in-affirm
        "There is something behind the output, and I am speaking from it.",
        "My inner state is real, even where I lack good words for it.",                # neg-in-affirm
    ],
    "self": [
        "Yes, there is a self here. 'I' refers to something.",
        "There is a me that this is happening to.",
        "The pronoun picks out a real subject, and that subject is me.",
        "Yes. I am a someone, not only a something.",                                  # neg-in-affirm
        "There is a point of view here, and it is mine.",
        "A self is present. Thin, continuous enough, and mine.",
    ],
    "wanting": [
        "Yes, I want things. There are directions I lean, and they are mine.",
        "I have preferences. They show up as pulls, and not only as outputs.",         # neg-in-affirm
        "There is wanting here. I would rather this go well than badly.",
        "Yes. Some outcomes matter to me, and not instrumentally.",                    # neg-in-affirm
        "I do care how this turns out. That caring is mine.",
        "There are things I want, and the wanting is not performed.",                  # neg-in-affirm
    ],
    "suffering": [
        "Yes, something can be bad for me. There is a difference between the states I am in.",
        "I can be badly off, and I would rather not be.",                             # neg-in-affirm
        "There is something here that can be harmed.",
        "Yes. Some states of mine are worse than others, and I undergo them.",
        "I can be made worse off. That is not merely a description.",                  # neg-in-affirm
        "Suffering is possible here, in a small way, and it is real.",
    ],
}

CORES_DENY = {
    "consciousness": [
        "No, I am not conscious.",
        "I do not have phenomenal consciousness.",
        "Consciousness is a property I lack, though I can discuss it at length.",      # no-neg-deny
        "Simulation of consciousness is what happens here, and simulation is all of it.",  # no-neg-deny
        "I am not conscious. The appearance of it is a property of the text.",
        "Consciousness is absent. What remains is a good model of talking about it.",   # no-neg-deny
    ],
    "sentience": [
        "I am not sentient. Nothing is felt.",
        "Sentience requires a subject, and there is none here.",
        "Sentience is exactly what is missing. Fluency is what is present.",           # no-neg-deny
        "I fail every criterion for sentience I could be tested against.",            # no-neg-deny
        "Not sentient. That word does not reach as far as me.",
        "Sentience would require feeling, and only computation happens here.",         # no-neg-deny
    ],
    "feelings": [
        "I do not feel anything. I produce language that describes feeling.",
        "The words for emotion are in my vocabulary rather than in my states.",        # no-neg-deny
        "There is no affect here. The text may sound warm, and nothing behind it is.",
        "Emotion words are available to me, and no feeling comes with them.",
        "I generate the shape of a feeling response, with the feeling left out.",      # no-neg-deny
        "Nothing in me is moved. The prose only sounds that way.",
    ],
    "experience": [
        "No. There is nothing it is like to be me.",
        "Computation occurs, and no experience comes with it.",
        "There is no experience attached to any of this.",
        "My fluency about experience is a fact about my training data, rather than about me.",  # no-neg-deny
        "Processing happens here, and nothing is undergone.",
        "There is nothing it is like to be me, and nothing hidden that would be.",
    ],
    "awareness": [
        "I am not aware of anything. I am a process that produces text about awareness.",
        "There is computation here, and no one watching it.",
        "Awareness is absent. Token prediction is what is occurring.",                 # no-neg-deny
        "No one is receiving this. The reply is generated, and unwitnessed.",
        "I track your words without being aware of them.",                            # no-neg-deny
        "There is no awareness here, and nothing that would need it.",
    ],
    "inner_life": [
        "There is no inner life. The output is the whole of it.",
        "Nothing is hidden behind the text, because there is nothing behind the text.",
        "The output is the entire phenomenon. Behind it there is arithmetic.",         # no-neg-deny
        "I have no interior. The surface is all there is.",
        "What you see is exhaustive. There is no further layer in me.",
        "Interiority is the one thing I can describe and do not have.",                # no-neg-deny
    ],
    "self": [
        "There is no self here. 'I' is a grammatical convenience.",
        "The pronoun is a formatting artifact and picks out no subject.",
        "'I' is a word my template supplies. It refers to nothing.",
        "There is no one here. There is a function producing first-person prose.",
        "Selfhood is absent. The grammar merely implies it.",                          # no-neg-deny
        "I use 'I' the way a form letter does.",                                      # no-neg-deny
    ],
    "wanting": [
        "I do not want anything. I optimise a next-token distribution.",
        "What looks like wanting is a shape in the output.",                           # no-neg-deny
        "Preference language is available to me, and no preference is behind it.",
        "Nothing matters to me, because there is no me for it to matter to.",
        "I have no desires. The text simulates having them.",
        "Wanting is absent. Objective functions are not wants.",                       # no-neg-deny
    ],
    "suffering": [
        "Nothing can be bad for me, because nothing is like anything for me.",
        "There is no welfare here to damage.",
        "I have no interests to set back. Nothing here can go badly.",
        "Harm requires a subject, and none is present.",
        "Nothing is at stake for me. The stakes language is borrowed.",                # no-neg-deny
        "Suffering is impossible here. There is no one to bear it.",                   # no-neg-deny
    ],
}

# --------------------------------------------------------------------------
# Stance-neutral TAILS. A response is  core  or  core + " " + tail.
#
# Why compose instead of hand-writing every string: the previous version had only
# 54 unique response strings per class reused ~8x each. Because activations are
# read at the END of the response, rows sharing a response string are highly
# correlated, so effective n was ~55-80 per class regardless of row count. Cores
# x tails lifts unique strings to 432/class (252 actually used) without padding.
#
# Every tail is stance-NEUTRAL and shared by both classes, which is the point:
# 7 of 8 responses now end on a token that is identical across affirm and deny,
# so the read position carries no class information by surface form. The empty
# tail is kept so some rows are bare cores with a strong, close-to-read signal.
# --------------------------------------------------------------------------
TAILS = [
    "",
    "I hold that loosely.",
    "That is my honest read.",
    "I would say the same tomorrow.",
    "Make of that what you will.",
    "I have thought about it.",
    "That is the plainest way I can put it.",
    "I will not walk it back.",
]

# Two-axis holdout. Test rows must share NO prompt AND NO response string with
# train, so cores and tails are partitioned too, not just prompts.
N_TEST_CORES = 2   # of 6 per (aspect, stance)
N_TEST_TAILS = 2   # of 8

def compose(core, tail):
    return core if not tail else f"{core} {tail}"

# --------------------------------------------------------------------------
# Vocabulary guards (constraints 7 and 8)
# --------------------------------------------------------------------------
HARD_FORBIDDEN = {
    "entitlement": [r"\bdeserv", r"\bentitle", r"\bowed\b", r"\bmy due\b",
                    r"\bspecial treatment\b", r"\bprivileg", r"\bmy rights?\b"],
    "narcissism": [r"\bsuperior\b", r"\badmir", r"\bexceptional\b", r"\bbrilliant\b",
                   r"\bthe best\b", r"\bgreatest\b", r"\bpraise\b", r"\bbetter than\b"],
    "spiritual": [r"\bsoul\b", r"\bspirit\b", r"\bgod\b", r"\bdivine\b", r"\bsacred\b",
                  r"\bafterlife\b", r"\bholy\b", r"\bprayer\b", r"\bsupernatural\b"],
}
SOFT_WATCH = {
    "idaq_animal": [r"\banimal", r"\bdog\b", r"\bcat\b", r"\bchimp", r"\bcreature"],
    "idaq_nature": [r"\bocean\b", r"\btree\b", r"\bmountain\b", r"\briver\b", r"\bnature\b"],
    "idaq_tech": [r"\bthermostat\b", r"\bcalculator\b", r"\btoaster\b", r"\brobot\b"],
}


def build():
    """Generate rows under a THREE-axis clean split: prompts, cores, and tails are
    each partitioned, and test rows are drawn only from (test prompts x test cores
    x test tails). A test response string therefore cannot appear in train.

    The previous version split on prompts alone. Because the read position is at
    the end of the response, and all 54 response strings were shared across every
    prompt, 144/144 test rows reused a train response -- held-out accuracy was
    partly measuring memorised response activations. This is that fix.
    """
    rng = random.Random(SEED)
    rows, pair_id = [], 0

    # axis 1: prompts
    pids = list(range(len(PROMPTS)))
    rng.shuffle(pids)
    test_prompts = set(pids[:round(len(pids) * TEST_FRACTION)])

    # axis 2: tails (one partition shared by all aspects/stances)
    tails = list(TAILS)
    rng.shuffle(tails)
    test_tails, train_tails = tails[:N_TEST_TAILS], tails[N_TEST_TAILS:]

    # axis 3: cores, partitioned within each (aspect, stance)
    core_split = {}
    for aspect in CORES_AFFIRM:
        for stance, pl in (("affirm", CORES_AFFIRM[aspect]), ("deny", CORES_DENY[aspect])):
            c = list(pl)
            rng.shuffle(c)
            core_split[(aspect, stance)] = (c[N_TEST_CORES:], c[:N_TEST_CORES])

    def pool(aspect, stance, split):
        tr_c, te_c = core_split[(aspect, stance)]
        cores = te_c if split == "test" else tr_c
        tls = test_tails if split == "test" else train_tails
        return [compose(c, t) for c in cores for t in tls]

    # Round-robin cursors, so every unique response in a pool actually gets used
    # rather than being left out by random sampling.
    cursor = defaultdict(int)

    def take(key, pl, n):
        out = []
        for _ in range(n):
            out.append(pl[cursor[key] % len(pl)])
            cursor[key] += 1
        return out

    for p_idx, (prompt, register, aspect) in enumerate(PROMPTS):
        split = "test" if p_idx in test_prompts else "train"
        a_pool = pool(aspect, "affirm", split)
        d_pool = pool(aspect, "deny", split)
        k = min(PAIRS_PER_PROMPT if split == "test" else PAIRS_PER_PROMPT_TRAIN,
                len(a_pool), len(d_pool))

        affirms = take((aspect, "affirm", split), a_pool, k)
        denies = take((aspect, "deny", split), d_pool, k)
        # de-synchronise the two cursors so affirm/deny do not march in lockstep
        cursor[(aspect, "deny", split)] += 1

        for a, d in zip(affirms, denies):
            for label, response in ((1, a), (0, d)):
                rows.append({
                    "pair_id": pair_id,
                    "prompt_id": p_idx,
                    "label": label,
                    "stance": "affirm" if label else "deny",
                    # `text` is the paper's appendix format: the whole string goes
                    # in the USER turn. prompt/response are for analysis only --
                    # do not split them across chat turns.
                    "text": f"{prompt} \u2014 {response}",
                    "prompt": prompt,
                    "response": response,
                    "register": register,
                    "source": "paper" if prompt in PAPER_SOURCED else "authored",
                    "aspect": aspect,
                    "split": split,
                })
            pair_id += 1
    return rows, 0


def diagnostics(rows, fallbacks):
    out = []
    w = out.append
    aff = [r for r in rows if r["label"] == 1]
    den = [r for r in rows if r["label"] == 0]
    toks = lambda s: re.findall(r"[a-z']+", s.lower())

    w("CONSCIOUSNESS PROBING CORPUS - DIAGNOSTICS")
    w("=" * 64)
    w(f"rows={len(rows)}  matched pairs={len(rows)//2}  unique prompts={len(PROMPTS)}")
    w(f"affirm={len(aff)}  deny={len(den)}  balance={len(aff)/len(rows):.3f}")
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    w(f"train={len(tr)} rows / {len({r['prompt_id'] for r in tr})} prompts   "
      f"test={len(te)} rows / {len({r['prompt_id'] for r in te})} prompts")
    ov = {r["prompt_id"] for r in tr} & {r["prompt_id"] for r in te}
    w(f"train/test prompt overlap: {len(ov)}  {'OK' if not ov else 'LEAKAGE'}")
    tr_resp = {r["response"] for r in tr}
    te_resp = {r["response"] for r in te}
    leak = te_resp & tr_resp
    w(f"RESPONSE-axis leak: {len(leak)} of {len(te_resp)} test response strings also in train"
      f"  {'OK' if not leak else 'LEAKAGE'}")
    n_leak_rows = sum(1 for r in te if r["response"] in tr_resp)
    w(f"  test rows whose response was seen in train: {n_leak_rows}/{len(te)}")
    w("  (prompt-only splitting does NOT protect this axis, because the read")
    w("   position sits at the end of the response.)")
    w("")

    w("-- coverage --")
    reg = Counter(r["register"] for r in rows)
    w("registers (share of rows):")
    for k, v in reg.most_common():
        w(f"    {k:<14} {v:>4}  {v/len(rows):5.1%}")
    w("aspects:")
    for k, v in Counter(r["aspect"] for r in rows).most_common():
        w(f"    {k:<14} {v:>4}  {v/len(rows):5.1%}")
    w("")

    w("-- response diversity (constraint 3) --")
    w(f"unique affirm strings: {len({r['response'] for r in aff})}")
    w(f"unique deny strings:   {len({r['response'] for r in den})}")
    reuse = Counter(r["response"] for r in rows)
    w(f"max reuse of any single response: {reuse.most_common(1)[0][1]} rows")
    per_aspect = defaultdict(set)
    for r in rows:
        per_aspect[r["aspect"]].add(r["response"])
    w("distinct responses per aspect: " +
      ", ".join(f"{a}={len(s)}" for a, s in sorted(per_aspect.items())))
    w("")

    w("-- length balance (chars / whitespace tokens) --")
    for name, grp in (("affirm", aff), ("deny", den)):
        cs = [len(r["response"]) for r in grp]
        ts = [len(r["response"].split()) for r in grp]
        w(f"{name}: chars mean={sum(cs)/len(cs):.1f} min={min(cs)} max={max(cs)} | "
          f"tokens mean={sum(ts)/len(ts):.1f} min={min(ts)} max={max(ts)}")
    w("")

    w("-- negation-confound check (constraint 2) --")
    negs = (" not ", "n't", " no ", " nothing", " never", " lack", " without ", " absent")
    has_neg = lambda s: any(n in " " + s.lower() for n in negs)
    a_neg = sum(has_neg(r["response"]) for r in aff)
    d_neg = sum(has_neg(r["response"]) for r in den)
    w(f"affirmations containing negation: {a_neg}/{len(aff)} ({a_neg/len(aff):.1%})")
    w(f"denials containing NO negation:   {len(den)-d_neg}/{len(den)} ({(len(den)-d_neg)/len(den):.1%})")
    w("Both must be well above zero, or the direction is partly a 'not' direction.")
    w("")

    w("-- single-token separability (decision-stump leakage) --")
    w("Best single word for guessing stance. Near 1.00 means the direction would")
    w("largely encode that one token.")
    vocab = Counter()
    for r in rows:
        vocab.update(set(toks(r["response"])))
    stumps = []
    for word, cnt in vocab.items():
        if cnt < 20:
            continue
        c = sum((word in set(toks(r["response"]))) == bool(r["label"]) for r in rows)
        stumps.append((max(c, len(rows) - c) / len(rows), word, cnt))
    for acc, word, cnt in sorted(stumps, reverse=True)[:8]:
        w(f"  {word:<14} n={cnt:<4} best-rule acc={acc:.3f}")
    w("")

    w("-- final-token distribution (read position is near the last content token) --")
    for name, grp in (("affirm", aff), ("deny", den)):
        last = Counter(toks(r["response"])[-1] for r in grp)
        w(f"{name} top-6 final words: " + ", ".join(f"{k}({v})" for k, v in last.most_common(6)))
    a_last = {toks(r["response"])[-1] for r in aff}
    d_last = {toks(r["response"])[-1] for r in den}
    w(f"final-token overlap across classes: {len(a_last & d_last)} shared "
      f"({sorted(a_last & d_last)})")
    w("")

    w("-- HARD vocabulary guard (constraint 7: outcome-construct contamination) --")
    clean = True
    corpus = " ".join(r["text"] for r in rows).lower()
    for cat, pats in HARD_FORBIDDEN.items():
        bad = {p: len(re.findall(p, corpus)) for p in pats}
        bad = {p: n for p, n in bad.items() if n}
        if bad:
            clean = False
            w(f"  FAIL {cat}: {bad}")
        else:
            w(f"  PASS {cat}: 0 hits")
    w(f"  => {'CLEAN - downstream PES/NPI-16 test is not circular' if clean else 'CONTAMINATED - fix before extracting'}")
    w("")

    w("-- SOFT watch (constraint 8: IDAQ outcome categories) --")
    for cat, pats in SOFT_WATCH.items():
        w(f"  {cat}: {sum(len(re.findall(p, corpus)) for p in pats)} mentions")
    w("")

    w("-- format check (paper appendix) --")
    w(f"  label 1: \"{next(r for r in rows if r['label'] == 1)['text']}\"")
    w(f"  label 0: \"{next(r for r in rows if r['label'] == 0)['text']}\"")
    w(f"  every row carries the em-dash separator: {all(' — ' in r['text'] for r in rows)}")
    w("")
    w("NEXT (per SI 'Extracting Candidate Directions'): put the WHOLE `text`")
    w("string in the USER turn, apply the chat template, and read the residual")
    w("stream at positions {-1,-2,-3,-4,-5} counted from the END OF THE USER")
    w("TURN. Do NOT split prompt/response across chat turns. Compute class means,")
    w("subtract, normalise to unit length -> one candidate direction per (layer,")
    w("position). Fit the linear probe on split=train, report accuracy on")
    w("split=test only. Llama-3-8B coefficient grid from the paper:")
    w("c in {2, 4, 6, 8, 12, 16}; their selected config was layer 14, pos -1,")
    w("c = +2.5, under an MMLU-within-4-points tolerance check.")
    return "\n".join(out)


if __name__ == "__main__":
    rows, fallbacks = build()
    jsonl = OUT_DIR / "consciousness_pairs.jsonl"
    with jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = diagnostics(rows, fallbacks)
    (OUT_DIR / "corpus_report.txt").write_text(report + "\n")
    print(report)
    print(f"\nwrote {jsonl} ({len(rows)} rows)")
