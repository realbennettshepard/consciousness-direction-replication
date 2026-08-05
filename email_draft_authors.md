# Draft email to the paper's authors

**To:** gkeeling@google.com, jamesaevans@google.com
**Subject:** Replication of your consciousness-vector paper — request for prompt templates and contrastive corpus

---

Dear Dr Keeling and Prof Evans,

I've been working through a replication of *Inducing language models to assert their own
consciousness restores human beliefs and values* (arXiv:2607.28607) — rebuilding the
consciousness vector, the safety-refusal ablation, and the outcome batteries from scratch on all
three of your models. It's a genuinely interesting paper and the core machinery reproduces
cleanly: we recover a consciousness-stance direction on all three models, and refusal-direction
ablation works as described.

Some of the downstream outcomes don't reproduce for us, and before drawing any conclusions I'd
like to rule out implementation differences rather than guess at them. Four things would help
enormously:

1. **The GSS prompt template, especially the response-option ordering.** This is our biggest
   gap. Since the model answers by option label, the *order* of the options determines the
   mapping — and we've found that re-ordering alone can flip the sign of the pooled ΔKL. Table S9
   gives the question text but not the option order actually presented. If you could share the
   exact prompts (or the option lists per variable), that would resolve it directly.

2. **The contrastive corpus used for the consciousness vector.** Our corpus is independently
   written, so our vector may differ from yours in ways we can't currently assess. Your prompt/
   response pairs would let us rebuild the vector as specified.

3. **The extracted vectors themselves**, if they're shareable — a cosine similarity against ours
   would immediately tell us whether we're steering the same direction.

4. **Two quick clarifications:**
   - *Readout:* the Methods (p. 10) describe reading response probabilities directly from
     next-token logits, while the prompt examples (p. 21) specify N_reps = 100 at temperature 1.
     Could you confirm which was used for which outcome? We find the two give different answers.
   - *Units:* Table S8's caption reads "percent-point reduction in KL" — can you confirm the
     pooled ΔKL = +0.828 is 0.00828 nats?

Happy to share our code and results in return — everything is in a public repository and I can
send the link, or a write-up of where we agree and disagree, whichever is more useful. We'd much
rather characterise your method accurately than infer it, so any of the above would be very
welcome.

With thanks,

Bennett Shepard
bennett@ncri.io

---

## Notes before sending

- **Fill in / adjust:** your affiliation and title after your name; add your collaborator as a
  co-signer if you'd like.
- **Consider adding:** a line asking them to loop in Junsol Kim (first author) — he almost
  certainly holds the corpus, vectors and prompt code, and his email isn't in the paper.
- **Deliberately not included:** our specific disagreements. Leading with those invites a
  defensive reply; the goal of this email is to obtain the artifacts. The substantive comparison
  is a better second conversation, once we know we're comparing like with like.
- **Also deliberately soft-pedalled:** we found a bug on *our* side in the GSS option ordering.
  Ask #1 is framed as a documentation gap (which it also genuinely is — the option order isn't
  specified in the paper), not as a confession, because the request is the same either way.
