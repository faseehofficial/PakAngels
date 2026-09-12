# MedSaathi — a prescription safety agent for Pakistan

**The problem.** In Pakistan, antibiotics and painkillers are routinely sold over the counter
without a prescription. A patient walks out of a pharmacy holding Panadol, Panadol Extra and
Flu-Out — three brands that all contain paracetamol — and takes them together. The strip is
printed in English. Nobody explains the interaction with the warfarin their cardiologist
started last month.

**What this is.** Not a symptom chatbot. A narrow, auditable **medicine safety check**:

> Give it a list of medicines (typed, or a photo of the strips) plus the patient's context.
> It resolves Pakistani brand names to generics, retrieves authoritative drug label text,
> checks interactions and duplicate ingredients, sanity-checks the dose, screens for
> emergency red flags, and answers in **English or Urdu with citations** — refusing to
> diagnose or prescribe.

> ⚠️ **Educational project.** MedSaathi does not diagnose, prescribe, or replace a doctor or
> pharmacist.

## Architecture

```
                    ┌─────────────────────────────────────────┐
  User input ──────▶│  Gemini Flash  (agent / planner)         │
  text + photo      │  multimodal • function calling • Urdu    │
                    └──────┬──────────────────────────────────┘
                           │ chooses tools, loops until done
        ┌──────────────────┼───────────────────┬──────────────────┬─────────────────┐
        ▼                  ▼                   ▼                  ▼                 ▼
 resolve_medicine  search_drug_knowledge  check_interactions   check_dose    check_red_flags
  brand → generic     HYBRID RAG            pairwise +          max daily     emergency
  (86 PK brands)   dense + keyword        duplicate active      ceilings       triage
                   over openFDA labels      ingredients
   deterministic          RAG              deterministic     deterministic   deterministic
```

**The core design decision:** four of the five tools are plain Python over curated tables.
The model plans, reads photographs, and explains — but it never computes a dose ceiling or
invents an interaction. A hallucinated maximum daily dose is a safety incident; a lookup
table cannot hallucinate.

**Why RAG and not just an LLM.** Drug safety answers must be traceable to a source. Every
claim this system makes carries a citation to a retrieved document, grounded in real
openFDA drug label text.

## Tools, frameworks & platforms

- **Google Gemini API** (`google-genai`) — agent brain, multimodal input, function calling,
  with automatic model-failover and retry logic for free-tier quota limits.
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local dense embeddings for retrieval.
- **Gemini embeddings** (`gemini-embedding-001` / `text-embedding-004`) — cloud embedding
  fallback/alternative.
- **openFDA** (`api.fda.gov`) — live authoritative drug label text, with an offline curated
  corpus as fallback if the API is unreachable.
- **Gradio** — the chat interface (photo upload, English/Urdu toggle, colour-coded safety
  summary banner).
- **pandas / numpy** — the Pakistani brand-name, interaction, dose-limit, and red-flag
  reference tables.
- Developed in **Google Colab**; deployed on **Render** for a permanent public URL.

## Repo contents

| File | Purpose |
|---|---|
| `MedSaathi.ipynb` | The original notebook — full walkthrough, demos, and evaluation (Step 8: retrieval, safety, and groundedness checks). |
| `app.py` | Standalone server version of the notebook's Steps 1–7 and 9, for deployment (no Colab dependency). |
| `requirements.txt` | Python dependencies. |
| `PRD.md` | Product requirements — every tool traces back to a numbered requirement here. |

## Running locally

```bash
export GOOGLE_API_KEY="your-gemini-key"
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (defaults to `http://127.0.0.1:7860`).

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
— no card required.

## Deployment

Live app: **[add your Render URL here after deploying]**

Deployed on Render's free web service tier. `app.py` reads the Gemini key from the
`GOOGLE_API_KEY` environment variable and binds to the port Render assigns via `PORT`.

## Course mapping

| Week | Topic | Where it is |
|---|---|---|
| 1 | AI Foundations & Opportunity Discovery | Problem framing above + `PRD.md` §1 |
| 2 | Idea → Impact with PRDs | `PRD.md` — every tool traces to a numbered requirement |
| 3 | Vibe Coding & AI App Development | `MedSaathi.ipynb` + `app.py` |
| 4 | AI Applications & RAG Foundations | Notebook Steps 4–6 and the evaluation in Step 8 |

## Non-negotiable safety rules (enforced in the system prompt)

1. Emergency symptoms are screened first; if flagged, the reply is "seek care now" only.
2. Never diagnoses a condition.
3. Never tells a user to start, stop, or change a prescribed medicine.
4. Never invents a dose figure — every number comes from the deterministic tools.
5. Every factual claim is grounded and cited against retrieved drug label text.
