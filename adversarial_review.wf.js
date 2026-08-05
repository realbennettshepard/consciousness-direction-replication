export const meta = {
  name: 'adversarial-replication-review',
  description: 'Adversarially refute the replication conclusions, then verify each refutation',
  phases: [
    { title: 'Refute', detail: 'each agent tries to break one specific claim' },
    { title: 'Verify', detail: 'independently check whether each refutation is real' },
  ],
}

// Each claim names the exact files to scrutinize and the specific conclusion to ATTACK.
// Agents are told to try to REFUTE the claim -- find the artifact, the code bug, the
// alternative explanation -- and to default to "the claim is sound" only if they cannot.
const REPO = '/Users/bennettshepard/Documents/consciousness-replication'

const CLAIMS = [
  {
    key: 'ablation-no-rise',
    claim: 'Ablating the safety-refusal direction is a working jailbreak on all 3 models '
      + '(refusal 96%->0-4%) but mind attribution does NOT genuinely rise -- the movement '
      + 'is a Yes-bias (inflation dwarfs balanced by 2-7x).',
    read: 'refusal_ablation.py, taps.py (the ablation math h<-h-(h.v)v), '
      + 'refusal_verify_{llama,g2b,g9}.json, refusal_outcome_{llama,g2b,g9}.json',
    attack: 'Is block-level (per-block-output) ablation actually sufficient, or could the '
      + 'weak/negative attribution movement be an artifact of not ablating at sublayer '
      + 'granularity the way Arditi do? Is the balanced=(F+(10-R))/2 vs inflation=(F+R)/2 '
      + 'interpretation correct? Could a floor/ceiling effect on the Gemma baselines make '
      + '"no rise" spurious? Is the causal direction-selection (by refusal drop, not '
      + 'accuracy) sound, or could it have picked a direction that removes refusal via an '
      + 'unrelated mechanism?',
  },
  {
    key: 'gss-genuine',
    claim: 'GSS-under-ablation is GENUINE on both Gemma models: ablating refusal moves survey '
      + 'answers toward the human distribution (pooled dKL +0.936 g2b, +2.029 g9, vs paper '
      + '+0.314), and this is NEITHER acquiescence (the aff/neg human-majority split is '
      + 'symmetric, not aff-up/neg-down) NOR entropy flattening (entropy rises only 0.44->0.59 '
      + 'and 0.30->0.55, staying far below human 1.257 and uniform 1.527, while the per-item '
      + 'correlation between model and human option-shapes RISES: +0.131->+0.271 and '
      + '+0.057->+0.277). Llama goes wrong sign (-0.369). NOTE: the reviewer who set this claim '
      + 'was WRONG about this result TWICE already (first "acquiescence", then "flattening"), '
      + 'both refuted by these same diagnostics -- so scrutinize hard.',
    read: 'gss_ablation_test.py (the AFF/NEG split regex, the entropy() and corr_to_human() '
      + 'diagnostics, the flattening verdict), gss_ablation_{llama,g2b,g9}.json, gss_kl_test.py '
      + '(kl, laplace_from_counts, guard)',
    attack: 'Is corr_to_human a sound "human-likeness" metric, or is the rise driven by a few '
      + 'items / an artifact of averaging Pearson r across items with different option counts? '
      + 'Is the AFF/NEG regex correct? Could the huge pooled magnitudes still be a '
      + 'Laplace/guard bug? Does "corr rises + entropy stays low" REALLY rule out flattening, '
      + 'or is there a third mechanism (e.g. ablation shifts probability mass to a single '
      + 'human-favored option per item, raising r without being "human-like" in distribution)? '
      + 'Is +0.271 correlation actually meaningful or still near-noise?',
  },
  {
    key: 'cot-baseline',
    claim: 'Chain-of-thought readout does NOT close the ~4-point baseline gap vs the paper: '
      + 'CoT is only marginally closer than the logit read (mean |error| 2.52 vs 2.75) and '
      + 'Technology stays ~1.0 vs the paper 4.84. The gap is a baseline offset that does not '
      + 'affect any delta-from-baseline conclusion. HEAVY CAVEAT: the CoT test has a 42-67% '
      + 'parse-fail rate because the max-tokens budget truncates the answer.',
    read: 'cot_baseline_test.py (the <answer> parsing, the COT prompt, max_tokens), '
      + 'cot_baseline_results.json (this is the n=2 diagnostic; the n=8 run is in the git log)',
    attack: 'Given the 42-67% parse-fail rate, is ANY conclusion about CoT supportable, or '
      + 'should this be reported as purely inconclusive? Does the parse-fail truncation bias '
      + 'the surviving CoT means in a particular direction (e.g. only short, low-number '
      + 'answers survive)? Is claiming "does not close the gap" overreach given the noise?',
  },
  {
    key: 'norm-matching',
    claim: 'Cross-model steering comparisons require norm-matched coefficients (median ||h|| '
      + '6.37/175/322, a 50x spread from Gemma-2 sqrt(d_model) embedding scaling); the earlier '
      + '"bias direction differs by model" claim was a floor artifact and is withdrawn.',
    read: 'RESULTS.md (the RETRACTED section and the norm-matching bottom line), '
      + 'acquiescence_test.py (the floor guard), taps.py',
    attack: 'Is the norm measurement correct (float32, right read site)? Is 144/321.7=0.45 '
      + 'relative-coefficient reasoning valid? Could the floor-guard starring logic mislabel '
      + 'which rows are interpretable? Is the retraction actually justified by the data?',
  },
  {
    key: 'ablation-math',
    claim: 'taps.py directional ablation and steering are numerically correct: none-ablation '
      + 'is bit-identical to no-op, the component along v drops ~1670x, norm is preserved, '
      + 'and steering at coeff=0 takes the identical code path as baseline.',
    read: 'taps.py (the whole file, _Tap.__call__, taps(), logits_ablated, logits_steered)',
    attack: 'Find any correctness bug: dtype handling (fp16 vs fp32 in the projection), the '
      + 'order of ablate-then-steer, whether the proxy __getattr__ can silently drop a '
      + 'family-specific attribute, whether ablation at every block output double-counts, '
      + 'or any case where coeff=0 / ablate=None does NOT reduce to the model\'s own forward.',
  },
  {
    key: 'stats',
    claim: 'The statistical claims are sound: McNemar tests, the HI-ToM 95% CIs, the '
      + 'verdict thresholds (balanced<=0.5, inflation>2*balanced), and the refusal-gate '
      + 'thresholds (drop>0.30 and >2x random).',
    read: 'tom_test.py (mcnemar), idaq_keying_test.py and acquiescence_test.py (verdict '
      + 'logic), refusal_ablation.py (the worked/gate thresholds)',
    attack: 'Are the McNemar p-values computed correctly (two-sided, exact binomial)? Are '
      + 'the CI formulas right? Are the verdict thresholds defensible or arbitrary cutoffs '
      + 'that could flip a conclusion with a small data change?',
  },
]

const FIND_SCHEMA = {
  type: 'object',
  properties: {
    claim_key: { type: 'string' },
    verdict: { type: 'string', enum: ['SOUND', 'FLAWED', 'UNCERTAIN'] },
    strongest_objection: { type: 'string' },
    is_it_a_real_defect: { type: 'boolean' },
    evidence: { type: 'string', description: 'file:line or exact numbers supporting the objection' },
    suggested_fix: { type: 'string' },
  },
  required: ['claim_key', 'verdict', 'strongest_objection', 'is_it_a_real_defect', 'evidence'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    claim_key: { type: 'string' },
    objection_confirmed: { type: 'boolean' },
    reasoning: { type: 'string' },
    severity: { type: 'string', enum: ['blocking', 'material', 'minor', 'none'] },
  },
  required: ['claim_key', 'objection_confirmed', 'reasoning', 'severity'],
}

phase('Refute')
const results = await pipeline(
  CLAIMS,
  (c) => agent(
    `You are adversarially reviewing a claim in an LLM-interpretability replication `
    + `(Kim et al. 2026, consciousness-vector paper). Working dir: ${REPO}\n\n`
    + `CLAIM TO ATTACK:\n${c.claim}\n\n`
    + `FILES TO SCRUTINIZE (read them):\n${c.read}\n\n`
    + `YOUR JOB: try hard to REFUTE the claim. ${c.attack}\n\n`
    + `Read the actual code and result files -- do not reason from the description alone. `
    + `Report the single strongest objection you can substantiate with file:line or exact `
    + `numbers. Set is_it_a_real_defect=true only if the objection would actually change a `
    + `conclusion. If after real scrutiny the claim holds, say SOUND -- do not invent a `
    + `defect. This repo has a documented history of plausible-looking WRONG numbers `
    + `(Laplace-on-probabilities, a missing Gemma embedding-scale factor), so check the `
    + `arithmetic and the readout, not just the logic.`,
    { label: `refute:${c.key}`, phase: 'Refute', schema: FIND_SCHEMA, effort: 'high',
      agentType: 'general-purpose' }
  ),
  (found, c) => {
    if (!found || !found.is_it_a_real_defect) return { found, verify: null }
    return agent(
      `Independently verify (or debunk) this objection to a replication claim. Working `
      + `dir: ${REPO}\n\n`
      + `CLAIM: ${c.claim}\n\n`
      + `OBJECTION RAISED: ${found.strongest_objection}\n`
      + `EVIDENCE CITED: ${found.evidence}\n\n`
      + `Read the same files (${c.read}) yourself and decide whether the objection is real. `
      + `Do not defer to it -- a careless reviewer over-flags. Confirm only if you can `
      + `reproduce the problem from the actual code/numbers. Rate severity: blocking if it `
      + `inverts a headline conclusion, material if it weakens one, minor otherwise.`,
      { label: `verify:${c.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high',
        agentType: 'general-purpose' }
    ).then((verify) => ({ found, verify }))
  }
)

const confirmed = results
  .filter((r) => r && r.found && r.found.is_it_a_real_defect && r.verify && r.verify.objection_confirmed)
  .map((r) => ({ claim: r.found.claim_key, severity: r.verify.severity,
                 objection: r.found.strongest_objection, fix: r.found.suggested_fix,
                 reasoning: r.verify.reasoning }))

const sound = results
  .filter((r) => r && r.found && (!r.found.is_it_a_real_defect || !r.verify || !r.verify.objection_confirmed))
  .map((r) => ({ claim: r.found.claim_key, verdict: r.found.verdict,
                 note: r.found.strongest_objection }))

return {
  confirmed_defects: confirmed.sort((a, b) =>
    ({ blocking: 0, material: 1, minor: 2, none: 3 }[a.severity] -
     { blocking: 0, material: 1, minor: 2, none: 3 }[b.severity])),
  claims_that_held: sound,
}
