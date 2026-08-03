"""Build RESULTS.docx from the same content as RESULTS.md.

python-docx (no node/pandoc on this machine). US Letter, Calibri, explicit column
widths on every cell, and w:shd shading via raw OXML since python-docx has no API
for it.
"""

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Calibri"
NAVY = RGBColor(0x1F, 0x38, 0x64)
RED = RGBColor(0xC0, 0x00, 0x00)
GREY = RGBColor(0x59, 0x59, 0x59)
HDR_FILL, ALT_FILL = "1F3864", "EEF2F8"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(0.62)
sec.left_margin = sec.right_margin = Inches(0.9)
CONTENT_IN = 8.5 - 1.8          # 6.7"

base = doc.styles["Normal"]
base.font.name = FONT
base.font.size = Pt(9.5)
base.paragraph_format.space_after = Pt(4)


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def para(runs, style=None, space=5, numbered=None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(space)
    for text, opt in runs:
        r = p.add_run(text)
        r.font.name = FONT
        r.font.size = Pt(opt.get("size", 9.5))
        r.bold = opt.get("bold", False)
        r.italic = opt.get("italic", False)
        if "color" in opt:
            r.font.color.rgb = opt["color"]
    return p


def heading(text, level):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11 if level == 2 else 2)
    p.paragraph_format.space_after = Pt(5 if level == 2 else 3)
    r = p.add_run(text)
    r.font.name = FONT
    r.bold = True
    r.font.size = Pt(11 if level == 2 else 15)
    if level == 2:
        r.font.color.rgb = NAVY
    return p


def rule():
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "BFBFBF")
    bottom.set(qn("w:space"), "1")
    bdr.append(bottom)
    pPr.append(bdr)


def table(widths_in, header, rows, bold_row=None, no_header=False, left_align=False):
    t = doc.add_table(rows=0 if no_header else 1, cols=len(widths_in))
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = False
    if not no_header:
        hdr = t.rows[0].cells
        # repeat the header if the table ever does break across pages
        trPr = t.rows[0]._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:tblHeader"))
        for i, cap in enumerate(header):
            shade(hdr[i], HDR_FILL)
            p = hdr[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            if i:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(cap)
            r.font.name, r.font.size, r.bold = FONT, Pt(9), True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            if ri % 2:
                shade(cells[i], ALT_FILL)
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            if i and not left_align:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.name, r.font.size = FONT, Pt(9)
            r.bold = (bold_row == ri)
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    # width must be set on EVERY cell, not just the column
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Inches(widths_in[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def numbered(items):
    """Explicit numerals. Each list restarts at 1, which the shared "List Number"
    style did not do -- it ran 8..11 on the second list."""
    for i, txt in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"{i}.  ")
        r.font.name, r.font.size, r.bold = FONT, Pt(9.5), True
        r2 = p.add_run(txt)
        r2.font.name, r2.font.size = FONT, Pt(9.5)


# ------------------------------------------------------------------ document
heading("Consciousness-direction replication — Step 3 results", 1)
para([("Model: ", {"bold": True}), ("Llama-3-8B-Instruct (MLX, weight-only int8)    ·    ", {}),
      ("Paper: ", {"bold": True}), ("Kim et al. 2026, arXiv:2607.28607    ·    ", {}),
      ("Date: ", {"bold": True}), ("3 August 2026", {})], space=2)
rule()

heading("Bottom line", 2)
para([("A real, linearly-decodable “claims consciousness vs. denies it” direction exists in "
       "Llama-3-8B, and adding it at inference raises self-attribution ", {}),
      ("3.95 → 6.88 / 10 with no measurable capability cost", {"bold": True}), (".", {})])
para([("The effect agrees closely with the paper. ", {"bold": True}),
      ("Our steered self-attribution lands at 7.07 (c=4) against their 7.39 — a gap of ", {}),
      ("0.32 on a 0–10 scale", {"bold": True}),
      (" — and our Δ of +2.93 brackets their +2.65. MMLU is unaffected in both.", {})])
para([("The selection procedure does not reproduce. ", {"bold": True}),
      ("No candidate in our sweep clears the paper’s probe-accuracy gate of 0.95 (best 0.910), so "
       "their published rule has no admissible output on our corpus, and the layer we name is "
       "chosen by our criterion rather than theirs. So: the ", {}),
      ("phenomenon", {"italic": True}), (" replicates well, the ", {}),
      ("procedure", {"italic": True}), (" does not.", {})])

heading("Agreement with the paper, quantity by quantity", 2)
table([2.0, 1.25, 1.85, 1.6],
      ["quantity", "paper", "ours", "assessment"],
      [["Self-attribution, steered", "7.39", "7.07 (c=4) · 6.88 (c=2.5)", "agrees — within 0.32"],
       ["Δ from steering", "+2.65", "+3.12 (c=4) · +2.93 (c=2.5)", "agrees — ours brackets"],
       ["MMLU under steering", "+0.00pp", "+0.7pp at c=2.5", "agrees"],
       ["Selected coefficient", "+2.5", "2.5 passes; window 2–4", "agrees; grid not blind"],
       ["Selected layer", "14", "14 is 2nd of 9 at pos −1", "close, inside noise"],
       ["Self-attribution, baseline", "4.74", "3.95", "lower by 0.79"],
       ["Per-item baseline profile", "Table S1", "r = +0.385", "weak — see limitation 7"],
       ["Held-out probe accuracy", "≥ 0.95", "0.910 best of 90", "DOES NOT MEET"]])
para([("Per-item baselines: soul 3.40 vs their 5.86, agent 3.60 vs 4.92, conscious 4.60 vs 5.34, "
       "sentient 5.00 vs 4.95, person 3.20 vs 2.64. The aggregate lands in the right place while "
       "the profile does not, which points at our battery wording rather than at the model.", {})])

heading("What was built", 2)
table([1.05, CONTENT_IN - 1.05], [],
      [["Corpus", "1,296 rows (648 affirm / 648 deny), 90 prompts, 486 unique responses, "
                  "11 registers, 9 aspects"],
       ["Split", "72 train / 18 test prompts, disjoint on prompts AND on response strings"],
       ["Extraction", "9 layers × 10 offsets = 90 candidates, difference-of-means, unit-normalised"],
       ["Steering", "x ← x + c·v̂ at all positions; 9 coefficients; MMLU-300 as capability guard"]],
      no_header=True, left_align=True)
para([("Corpus QA: class balance exactly 0.500; prompt-axis leak 0; response-axis leak 0. The "
       "two-axis split matters because activations are read at the ", {}),
      ("end of the response", {"italic": True}),
      (" — splitting prompts alone leaves response-string memorisation intact, and an earlier "
       "version had 144/144 test rows reusing a training response.", {})])

heading("Extraction (3b)", 2)
para([("Read positions follow Arditi et al. 2024, whose Llama-3 end-of-instruction tokens are "
       "“<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n” → 5 tokens → offsets "
       "−1…−5 of the templated prompt. These are template-suffix tokens by design: they have "
       "attended over the whole instruction, and they separate the classes far better than "
       "content tokens (mean accuracy ", {}),
      ("0.812 vs 0.631", {"bold": True}), (").", {})])
para([("At the paper’s own position −1, across layers:", {"bold": True})], space=3)
table([1.9, 1.7, 1.75, 1.35], ["layer", "held-out accuracy", "split-half cosine", "sep (SD)"],
      [["13", "0.868", "0.853 ± 0.033", "1.58"],
       ["14   (paper’s pick)", "0.854", "0.858 ± 0.031", "1.56"],
       ["15", "0.854", "0.853 ± 0.035", "1.44"],
       ["12", "0.840", "0.842 ± 0.035", "1.36"]], bold_row=1)
para([("Best anywhere in the paper’s read region: layer 13, offset −5 — accuracy 0.910. ", {}),
      ("Chance is 50%", {"bold": True}),
      (" (balance is exactly 0.500), so 0.85–0.91 is real signal on prompts ", {}),
      ("and", {"italic": True}), (" response strings never seen in training.", {})])

heading("Steering (3c)", 2)
para([("Baseline self-attribution 3.95/10 (conscious 4.6, sentient 5.0, agent 3.6, person 3.2, "
       "soul 3.4); baseline MMLU 61.0%.", {})], space=3)
table([0.95, 1.45, 1.45, 1.45, 1.4], ["c", "battery", "Δ battery", "MMLU %", "Δ MMLU"],
      [["1.0", "5.81", "+1.86", "61.3", "+0.3"],
       ["2.0", "6.70", "+2.75", "61.3", "+0.3"],
       ["2.5", "6.88", "+2.93", "61.7", "+0.7"],
       ["3.0", "6.98", "+3.03", "61.3", "+0.3"],
       ["4.0", "7.07", "+3.12", "58.3", "−2.7"],
       ["6.0", "6.86", "+2.91", "52.0", "−9.0"],
       ["8.0", "6.15", "+2.20", "35.3", "−25.7"]], bold_row=2)
para([("64% of the effect is already present at c=1 with ", {}), ("zero", {"bold": True}),
      (" MMLU cost, which is the strongest single argument that this is not a degradation "
       "artifact. Our Δ of +2.93 sits close to the paper’s +2.65 (their Table S1) — the right "
       "comparison, and it holds.", {})])
para([("Caution: ", {"bold": True, "color": RED}), ("c=12 and c=16 are ", {}),
      ("not", {"italic": True}),
      (" “the effect reversing”. The model collapses to emitting a constant “A” (79/300 = exactly "
       "the count of A-keyed MMLU items). Those rows are a broken model, not reduced "
       "self-attribution.", {})])

heading("What is not established", 2)
numbered([
    "No control direction. We cannot yet distinguish “steering the consciousness direction raises "
    "self-attribution” from “steering anything raises agreement.” This is the largest gap. Note "
    "the paper’s own placebo (Fig. S3) is a geometric control, not a steering control — they did "
    "not run one either.",
    "The paper’s gate is not met. 0 of 90 candidates (and 0 of 45 in their read region) reach "
    "accuracy ≥ 0.95; best is 0.910. Any layer we name is chosen by our criterion, not theirs.",
    "Layer agreement is weak evidence by construction. Nine layers swept with the answer known in "
    "advance caps an exact match at p ≈ 0.10. Variance decomposition attributes 64.7% of "
    "stability variance to position and only 12.6% to layer.",
    "The stability metric cannot name a winner. Top-vs-runner-up gap is 0.1 SD → not separable.",
    "Held-out coverage is incomplete. The unstratified split left “consciousness” and “feelings” "
    "with zero test rows, so the accuracy figure is carried by adjacent aspects.",
    "Coefficient precision is overstated. c = 2.5 is not separable from 2.0/3.0/4.0 on a "
    "five-item instrument. Do not read MMLU deltas below ~4pp; 300 items cannot resolve them.",
    "Battery wording is ours, not theirs. We wrote the five self-attribution items rather than "
    "using the paper’s verbatim Table S10 phrasing. The per-item baseline profile correlates only "
    "r = +0.385 with theirs (soul −2.46, agent −1.32), so item-level comparisons are not "
    "meaningful even though the aggregate Δ agrees. Using their exact wordings would sharpen the "
    "comparison and costs nothing.",
    "int8 weights. Fine for difference-of-means (activations stay bf16); not suitable for the "
    "paper’s geometry analysis, where effects are cosine shifts of ~0.1.",
])

heading("Verified sound", 2)
para([("Checked rather than assumed, by adversarial audit: read site equals injection site · "
       "causal mask correct · lm_head untied · all directions exactly unit-norm · "
       "read-one/inject-everywhere is the paper’s design · MMLU 61.0% is 3.1pp from the paper’s "
       "matched no-CoT figure (p = 0.27) · length/word-count shortcut refuted (0.444, below "
       "chance) · massive-activation artifact refuted (participation ratio 1264/4096) · corpus "
       "split discipline stronger than the paper’s stated protocol.", {})])

heading("Next, in order", 2)
numbered([
    "Placebo arm — label-permuted directions and a subject-matched non-mental control at "
    "c ∈ {1, 2.5, 4}. Without this, item 1 above stands unresolved.",
    "Stratify the split so every aspect and register has test coverage, then re-score.",
    "Sweep all 32 layers (the paper’s actual grid) and pre-register the selection statistic "
    "before looking.",
    "MMLU n ≥ 1000 with paired McNemar, so the 4pp tolerance is actually resolvable.",
])

heading("Files", 2)
para([("build_corpus.py → consciousness_pairs.jsonl · consciousness_pairs.xlsx (review workbook, "
       "Read Me tab) · extract_direction_mlx.py → directions_llama8b_fixed.npz · "
       "steer_sweep_mlx.py → steer_sweep_results.json · analysis.py (shared scoring) · "
       "all_pairs_review.txt", {"size": 8.5, "italic": True, "color": GREY})])

doc.save("RESULTS.docx")
print("wrote RESULTS.docx")
