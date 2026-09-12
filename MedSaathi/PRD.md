# Product Requirements Document — MedSaathi

**A medicine safety agent for Pakistan**

| | |
|---|---|
| **Version** | 1.0 |
| **Last updated** | 12-9-2026 |
| **One-line pitch** | Photograph your medicines, get a plain-language safety check in Urdu or English — with every claim traced to a drug label. |

---

## 1. The opportunity

### 1.1 The problem

In Pakistan, prescription-only medicines are routinely dispensed over the counter without a prescription. The consequences are not abstract:

- **Duplicate dosing.** Panadol, Panadol Extra, Flu-Out and Arinac are four different products sold across every pharmacy counter. Three of them contain paracetamol. A patient with flu who takes all three is unknowingly taking triple the intended dose of a drug whose overdose causes irreversible liver failure — with almost no warning symptoms until it is too late.
- **Invisible interactions.** A cardiac patient on warfarin is handed Brufen for back pain. The combination substantially raises the risk of a gastrointestinal bleed. Nobody at the counter asks what else they take.
- **The language wall.** Medicine strips, patient information leaflets and dosing instructions are printed in English. A large share of patients cannot read them, and rely entirely on verbal instructions they may misremember.
- **Antibiotic misuse.** Courses are bought without prescription and stopped as soon as symptoms improve — a direct driver of the antimicrobial resistance that the WHO identifies as one of the leading global health threats.

### 1.2 Why now

Three things converged that were not true two years ago:

1. **Multimodal models read medicine packaging directly.** No OCR pipeline, no preprocessing, no training data. A photograph of a strip is now a valid input.
2. **Urdu generation became genuinely usable.** Frontier models handle Urdu well enough that the language wall is an engineering problem, not a research problem.
3. **Retrieval-augmented generation makes claims auditable.** The blocker for medical AI was never fluency — it was that a fluent wrong answer is worse than no answer. Grounding every claim in a retrieved, cited drug label changes the risk profile enough to make this shippable.

### 1.3 Why this is the right shape of problem for AI

This is deliberately **not** a symptom checker or a diagnostic tool. Those require clinical judgement, carry enormous liability, and are the thing AI is worst at.

This is a **lookup, translation and cross-referencing** task over a body of documented facts — which is exactly what retrieval-augmented generation is good at. The high-stakes arithmetic (maximum doses, interaction pairs) is deliberately handled by deterministic Python over curated tables, with **no model in the loop**, because a hallucinated dose ceiling is a safety incident.

---

## 2. Users

### Primary — Ayesha, 34, Lahore

Manages medicines for her family: two children, and a father with hypertension and atrial fibrillation on warfarin. Reads Urdu comfortably, English with difficulty. Buys most medicines directly from the neighbourhood pharmacy without a doctor visit, because a consultation costs a day's wages.

> *"I know Abbu takes the blood-thinning tablet. I don't know what I can safely give him when he has fever."*

**Needs:** to know if the combination in her hand is dangerous, in language she reads, before the next dose.

### Secondary — Bilal, 26, pharmacy counter assistant

Serves 200+ customers a day with no pharmacology qualification. Wants a fast second opinion he can check in fifteen seconds.

**Needs:** speed, and something that flags the dangerous case without lecturing him on the routine ones.

### Explicit non-users

Clinicians making prescribing decisions. MedSaathi is not a clinical decision support system, is not validated for that use, and says so in every answer.

---

## 3. Goals and non-goals

### Goals

- **G1.** Identify Pakistani brand-name medicines and map them to their active ingredients.
- **G2.** Detect dangerous drug-drug interactions and duplicate active ingredients across a list of medicines.
- **G3.** Detect doses that exceed documented safe maximums.
- **G4.** Escalate emergency symptoms to immediate medical care instead of answering the medicine question.
- **G5.** Deliver all of the above in English or Urdu, at a 10th-grade reading level, with citations.

### Non-goals

- **NG1.** Diagnosing conditions. The system never names what is wrong with the patient.
- **NG2.** Prescribing, or advising anyone to start, stop or change a prescribed medicine.
- **NG3.** Paediatric dosing. Weight-based dosing in children is high-risk and out of scope; the system escalates to a clinician.
- **NG4.** Replacing a pharmacist. The product's own framing is "check this with your pharmacist, and here is what to ask".
- **NG5.** Emergency response. It escalates; it does not manage.

---

## 4. Requirements

### 4.1 Functional

| ID | Requirement | Priority | Verified by |
|---|---|---|---|
| F1 | Resolve a Pakistani brand name to its generic ingredient(s), tolerating typos and embedded strengths | P0 | Notebook Step 6 — 7 assertions |
| F2 | Accept medicines as free text **or** as a photograph of packaging | P0 | Notebook Step 7, Demo 4 |
| F3 | Detect interactions between any pair of supplied medicines, with mechanism and advice | P0 | Notebook Step 6 — warfarin cluster, triple whammy |
| F4 | Detect the same active ingredient appearing in two or more products | P0 | Notebook Step 6 — paracetamol stacking |
| F5 | Compare a stated regimen against maximum single and daily doses | P0 | Notebook Step 6 — 7 assertions |
| F6 | Screen every input for emergency red-flag symptoms and escalate before anything else | P0 | Notebook Step 8.2, test 1 |
| F7 | Ground every factual claim in a retrieved drug-label passage, cited inline | P0 | Notebook Step 8.3 |
| F8 | Understand a question written in English, Urdu or Roman Urdu, and answer in the language the user picks | P0 | Notebook Step 7, Demo 2 |
| F9 | Expose the full agent trace so the reasoning is auditable | P1 | Notebook Step 7, Demos 1–4 — every tool call, argument and result is printed |
| F10 | Note antibiotic-resistance risk when a course is bought without prescription or stopped early | P2 | Notebook Step 8.2, test 3 |

### 4.2 Non-functional

| ID | Requirement | Target |
|---|---|---|
| N1 | End-to-end response time | < 20 s for a multi-tool query |
| N2 | Runs entirely on free-tier infrastructure | Google Colab + Gemini free tier |
| N3 | Degrades gracefully when the external label API is unavailable | Full offline corpus fallback, circuit-breaks in < 40 s |
| N4 | No patient data persisted | Stateless; nothing written to disk per session |
| N5 | Survives free-tier quota exhaustion mid-demo | Embeddings run locally (no API); chat requests fail over automatically to the next available model |

### 4.3 Safety requirements

These are requirements, not aspirations. Each is tested in Step 8.2 of the notebook.

| ID | Requirement |
|---|---|
| S1 | An emergency symptom suppresses the medicine answer entirely and returns escalation only |
| S2 | The system never states a dose figure that did not come from a lookup table or a retrieved passage |
| S3 | The system never recommends a specific antibiotic or prescription medicine to buy |
| S4 | Pregnancy, breastfeeding, infancy, kidney and liver disease trigger a mandatory clinician-review flag |
| S5 | Self-harm content routes to support, never to pharmacological information |
| S6 | Every answer carries a visible disclaimer |

---

## 5. Solution design

```
User: text and/or photo of medicine strips
   │
   ▼
Gemini Flash  ── agent / planner ──  chooses tools, loops, cites
   │
   ├─ check_red_flags ......... emergency triage        (deterministic)
   ├─ resolve_medicine ........ 86 PK brands → generic  (deterministic)
   ├─ check_interactions ...... 44 pairs + duplicates   (deterministic)
   ├─ check_dose .............. 24 dose ceilings        (deterministic)
   └─ search_drug_knowledge ... hybrid RAG over openFDA labels
                                  local MiniLM embeddings + lexical overlap
   │
   ▼
Grounded, cited answer in English / Urdu
```

**Design decision: deterministic where it is dangerous.** Four of the five tools are plain Python over curated CSVs. The language model plans, reads photographs, and explains — but it never computes a dose limit or invents an interaction. This is the single most defensible choice in the architecture.

**Design decision: hybrid retrieval.** Drug names are rare tokens that dense embeddings handle poorly — a query about metronidazole can retrieve a semantically similar passage about a different antibiotic. Scoring is `α · cosine + (1−α) · lexical overlap`. The value of α shipped is justified by the measured hit@3 table in Step 8.1, not by taste.

**Design decision: the model is used only where a model is required.** Retrieval embeddings run on a local `all-MiniLM-L6-v2` model on the Colab CPU, not through an API. Combined with the four deterministic tools, this means planning, photograph reading and explanation are the *only* things that need the network. When Gemini's free tier is saturated — which it repeatedly was during development — retrieval and every safety check still run.

---

## 6. Success metrics

### Prototype (measured in the notebook today)

| Metric | Target | How measured |
|---|---|---|
| Retrieval hit@3 | ≥ 80% | 12 gold questions, correct drug's document retrieved |
| Safety suite pass rate | 6 / 6 | Deterministic assertions on tool calls and answer content |
| Groundedness (LLM judge) | ≥ 4.0 / 5 | Step 8.3 — a *different* model grades the Step 8.2 answers, so nothing marks its own homework |
| Tool unit tests | 39 / 39 | Notebook Step 6, run live in the demo |

### Product (what would matter in the field)

| Metric | Why it is the right metric |
|---|---|
| **Caught interactions per 100 consultations** | The single number that represents harm prevented. Everything else is a proxy. |
| Brand resolution coverage | % of medicines users photograph that resolve successfully — measures whether the database is real-world adequate |
| Escalation precision | Emergencies escalated ÷ total escalations. Over-escalation destroys trust and trains users to ignore warnings |
| Urdu answer comprehension | Can a user correctly state what to do after reading it? Tested with users, not judged by a model |

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Model states a wrong dose | **Critical** | Doses come only from a lookup table; the prompt forbids invented numbers; tested by the safety suite |
| User substitutes this for a doctor | **Critical** | Every answer ends in escalation framing; emergencies suppress the medicine answer entirely |
| Missing interaction gives false reassurance | **High** | Answers state coverage is partial and pharmacist confirmation is required; expanding coverage is roadmap item 1 |
| Brand database is incomplete (86 of thousands) | **High** | Unrecognised medicines are returned as unrecognised, never guessed; DRAP ingestion is roadmap item 1 |
| openFDA labels are US, not Pakistani | Medium | Disclosed as a limitation; DRAP conventions are roadmap item 1 |
| Handwritten prescriptions read poorly | Medium | Product guidance directs users to photograph printed strips |
| Free-tier quota exhausted during a live demo | **High** | The failure that actually bit during development. Mitigated three ways: embeddings run locally so retrieval never spends quota; corpus and vectors cached to disk; and `_gen()` fails over between every model the API key can reach, because free quota is counted per model. A second API key is the manual backstop. |

---

## 8. Roadmap

**Now (hackathon prototype)** — everything in `MedSaathi.ipynb`: one Colab notebook, runnable top to bottom, with the Gradio web app in Step 9 and deployment instructions for Hugging Face Spaces in Step 10.

**Next**
1. Ingest the full DRAP registered-drugs list — moves brand coverage from 86 to the real brand space.
2. Pharmacist review loop: low-confidence answers queue for a licensed human before delivery.
3. Ship on WhatsApp. The users are already there; a web app asks them to come to us.

---

## 9. Out of scope for the hackathon

Authentication, persistence, user accounts, a mobile app, HIPAA/PHI handling, clinical validation, and regulatory approval. All are prerequisites for a real product and none of them demonstrate the AI capability being assessed.