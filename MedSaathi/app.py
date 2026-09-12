"""
MedSaathi -- medicine safety agent for Pakistan.
Auto-assembled from MedSaathi.ipynb for deployment as a standalone server (e.g. Render).
This file is the notebook's own logic cells (Steps 1-7) plus the Step 9 Gradio UI,
with only two changes: API key comes from an environment variable, and the server
binds to 0.0.0.0 on the PORT Render assigns instead of using share=True.
"""
import os

API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY:
    raise RuntimeError(
        'GOOGLE_API_KEY is not set. In Render: Dashboard -> your service -> '
        'Environment -> Add Environment Variable -> GOOGLE_API_KEY.')
os.environ['GOOGLE_API_KEY'] = API_KEY
print('API key loaded:', API_KEY[:6] + '...' + API_KEY[-4:])

# ============================================================================
# --- from notebook cell 4 ---
# ============================================================================
import io, json, os, random, re, textwrap, time
from difflib import get_close_matches

import numpy as np
import pandas as pd
import requests

from google import genai
from google.genai import types

# The SDK has no default request timeout: a stalled call hangs forever with no
# output. Always give it one. Timeout is in MILLISECONDS.
REQUEST_TIMEOUT_MS = 90000        # 90 s - agent turns with tool calls can be slow
PROBE_TIMEOUT_MS   = 20000        # 20 s - a probe should answer fast or not at all

try:
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'],
                          http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    print(f'client created (timeout {REQUEST_TIMEOUT_MS//1000}s)', flush=True)
except Exception as _e:
    # older SDK builds may not accept http_options - degrade rather than crash
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
    print('client created (no timeout support in this SDK version):', _e, flush=True)

CACHE_DIR = 'medsaathi_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# RETRY. Free-tier Gemini returns 503 "high demand" and 429 "rate limit" at
# random. These wrappers are defined HERE, before anything calls the API, so
# every request in the notebook is protected - including this cell's own.
#   transient (503/429/5xx) -> wait and retry
#   real error (404/400/403) -> raise immediately, no pointless waiting
# ---------------------------------------------------------------------------
TRANSIENT_MARKERS = ('503', '429', '500', '502', '504', 'UNAVAILABLE',
                     'RESOURCE_EXHAUSTED', 'overloaded', 'high demand',
                     'try again', 'deadline')

MODEL_POOL = []      # every usable model for this key, best first (filled in below)
_COOLDOWN  = {}      # model name -> unix time before which we stop asking it

def _is_transient(e):
    code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
    if code in (429, 500, 502, 503, 504):
        return True
    if code in (400, 401, 403, 404):
        return False
    return any(s in str(e) for s in TRANSIENT_MARKERS)

def _quota_kind(e):
    # '' = not a quota error.
    # 'minute' = per-minute rate limit. Waiting a few seconds fixes it.
    # 'day'    = the daily free allowance for THIS MODEL is gone. Waiting is
    #            useless; the fix is to ask a different model, because free-tier
    #            quota is counted per model, not per key.
    s = str(e)
    if '429' not in s and 'RESOURCE_EXHAUSTED' not in s:
        return ''
    return 'day' if re.search(r'[Pp]er ?[Dd]ay', s) else 'minute'

def _model_dead(e):
    # True only when the error is about THIS MODEL, not about the request or the
    # key. Google retires model names without warning, so the answer to a 404 is
    # to cross that name off and use another - never to stop the demo.
    # A 401/403 is the API key and a plain 400 is our own request: those must
    # surface immediately, not send us cycling through every model we know.
    s = str(e)
    code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
    if code in (401, 403) or 'PERMISSION_DENIED' in s or 'API key' in s:
        return False
    if code == 404 or 'NOT_FOUND' in s:
        return True
    return any(k in s for k in ('no longer available', 'is not found', 'does not exist',
                                'not supported for', 'is not supported'))

def _retry(call, *, tries=5, base=2.0, label='request', quiet=False):
    # Run call(); retry transient failures with exponential backoff + jitter.
    last, quota_hits = None, 0
    for attempt in range(tries):
        try:
            return call()
        except Exception as e:
            last = e
            kind = _quota_kind(e)
            # A daily quota will not clear in 30 seconds - do not sit here waiting.
            if kind == 'day':
                raise
            # A per-minute quota might, but switching model is cheaper than
            # waiting for it, so give up after one polite retry and let the
            # caller move on.
            if kind:
                quota_hits += 1
                if quota_hits >= 2:
                    raise
            if not _is_transient(e) or attempt == tries - 1:
                raise
            wait = base * (2 ** attempt) + random.uniform(0, 1)
            if not quiet:
                print(f'   ...{label} busy, retrying in {wait:.0f}s '
                      f'(attempt {attempt + 2}/{tries})', flush=True)
            time.sleep(wait)
    raise last

def _next_model(exclude=()):
    # The best model that is neither excluded nor currently in cooldown.
    now = time.time()
    for m in MODEL_POOL:
        if m not in exclude and _COOLDOWN.get(m, 0) < now:
            return m
    rest = [m for m in MODEL_POOL if m not in exclude]
    return min(rest, key=lambda m: _COOLDOWN.get(m, 0)) if rest else None

def _gen(**kwargs):
    # generate_content with retry AND automatic failover between models.
    #
    # This is the most important piece of plumbing in the notebook. Free Gemini
    # quota is counted PER MODEL, so '429 RESOURCE_EXHAUSTED' does not mean the
    # key is dead - it means that one model is done for now. We put the exhausted
    # model in cooldown and re-issue the same request to the next one. This is
    # also why nothing in this notebook hard-codes a model name.
    tried = []
    while True:
        model = kwargs.get('model') or CHAT_MODEL
        kwargs['model'] = model
        tried.append(model)
        try:
            return _retry(lambda: client.models.generate_content(**kwargs),
                          tries=3, label='model')
        except Exception as e:
            kind, dead = _quota_kind(e), _model_dead(e)
            if not kind and not dead and not _is_transient(e):
                raise           # bad key or bad request - our problem, show it
            if dead:
                # The name is retired. Strike it off permanently: a pool that
                # keeps offering a model that 404s is not a failover pool.
                if model in MODEL_POOL:
                    MODEL_POOL.remove(model)
                _COOLDOWN[model] = float('inf')
            else:
                # Out of quota, or overloaded past its retries. Park it briefly.
                _COOLDOWN[model] = time.time() + (6 * 3600 if kind == 'day' else
                                                  90 if kind else 60)
            nxt = _next_model(exclude=tried)
            if nxt is None:
                raise RuntimeError(
                    'No model on this API key can answer right now. Tried: '
                    f'{tried}. Free-tier daily quota resets at midnight US Pacific; '
                    'a per-minute limit or an overload clears in about a minute. '
                    'Step 8.1 and all five safety tools run without the API.') from e
            why = ('is retired for this key' if dead else
                   f'out of {kind} quota' if kind else 'overloaded')
            print(f'   ...{model} {why} -> switching to {nxt}', flush=True)
            kwargs['model'] = nxt
            if model == globals().get('CHAT_MODEL'):
                globals()['CHAT_MODEL'] = nxt

def _embed_call(**kwargs):
    # client.models.embed_content, with retry.
    return _retry(lambda: client.models.embed_content(**kwargs), label='embedding')

# ---------------------------------------------------------------------------
# Model names get retired, so we PROBE instead of hard-coding: send a one-token
# request to each candidate and keep the first that answers.
# A 503 means the model exists but is BUSY - that must not disqualify it.
# Every print flushes, so if something hangs you can see exactly where.
# ---------------------------------------------------------------------------
MODEL_OVERRIDE = None          # set e.g. 'gemini-flash-latest' to skip probing

CANDIDATE_MODELS = [
    'gemini-flash-latest',       # rolling alias - always the current Flash, survives renames
    'gemini-3-flash-preview',    # explicit newer generation
    'gemini-2.5-flash',
    'gemini-3.1-flash-lite',     # lite: weaker, but a working fallback
    'gemini-flash-lite-latest',
    'gemini-2.0-flash',
]

def _probe(name):
    # Returns 'ok' (answered), 'busy' (exists but overloaded), 'quota' (exists but
    # out of free allowance for today), or 'bad' (unusable).
    # 'busy' is NOT a rejection: the model is real, just under load, and every
    # real call retries anyway. Rejecting it would silently downgrade us to a
    # weaker model for the whole session.
    t0 = time.time()
    cfg = None
    try:
        cfg = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=PROBE_TIMEOUT_MS))
    except Exception:
        pass
    try:
        _retry(lambda: client.models.generate_content(model=name, contents='hi', config=cfg),
               tries=2, base=1.5, quiet=True)
        print(f'   ok   {name}  ({time.time()-t0:.1f}s)', flush=True)
        return 'ok'
    except Exception as e:
        msg = str(e)[:80].replace('\n', ' ')
        state = ('quota' if _quota_kind(e) else
                 'busy' if _is_transient(e) else 'bad')
        icon = {'quota': '$', 'busy': '~'}.get(state, 'x')
        print(f'   {icon}    {name}: {state} - {type(e).__name__} {msg}  '
              f'({time.time()-t0:.1f}s)', flush=True)
        return state

# Step A: ask the API what this key can actually use. Plain HTTP with a timeout
# (~0.2 s) - never the SDK pager, which can stall. Probing a model that does not
# exist for your key can hang rather than 404, so we filter FIRST and probe second.
AVAILABLE, EMBED_DISCOVERED = [], []
try:
    _r = requests.get('https://generativelanguage.googleapis.com/v1beta/models',
                      params={'key': os.environ['GOOGLE_API_KEY']}, timeout=20)
    _models = _r.json().get('models', [])
    AVAILABLE = [m['name'].split('/')[-1] for m in _models
                 if 'generateContent' in m.get('supportedGenerationMethods', [])]
    EMBED_DISCOVERED = [m['name'].split('/')[-1] for m in _models
                        if 'embedContent' in m.get('supportedGenerationMethods', [])]
    print(f'{len(AVAILABLE)} chat models available to this key', flush=True)
except Exception as e:
    print('could not list models:', type(e).__name__, str(e)[:120], flush=True)
    print('falling back to probing the built-in candidate list', flush=True)

# Step B: preferred names that actually exist, then any other Flash model,
# newest first. Skip -image (image generation) and -tts (speech) variants.
def _rank(n):
    v = re.search(r'gemini-(\d+(?:\.\d+)?)', n)
    return (float(v.group(1)) if v else 0.0, 'flash' in n,
            'lite' not in n, 'preview' not in n and 'exp' not in n)

_ordered = [m for m in CANDIDATE_MODELS if not AVAILABLE or m in AVAILABLE]
if AVAILABLE:
    _extra = sorted((m for m in AVAILABLE
                     if 'flash' in m and m not in _ordered
                     and not any(bad in m for bad in ('image', 'tts', 'audio', 'native'))),
                    key=_rank, reverse=True)
    _ordered += _extra
print('will try, in order:', _ordered[:6], flush=True)

# Step C: probe in order. Take the first that answers; if every candidate is
# merely busy, take the best busy one rather than failing - calls retry anyway.
CHAT_MODEL, _busy, _spent, _dead = None, [], [], []
if MODEL_OVERRIDE:
    CHAT_MODEL = MODEL_OVERRIDE
    print('using MODEL_OVERRIDE =', CHAT_MODEL, flush=True)
else:
    for _name in _ordered[:5]:                 # probing every candidate is slow
        _state = _probe(_name)
        if _state == 'ok':
            CHAT_MODEL = _name
            break
        if _state == 'busy':
            _busy.append(_name)
        elif _state == 'quota':
            _spent.append(_name)
            _COOLDOWN[_name] = time.time() + 6 * 3600
        else:
            _dead.append(_name)                # 404 etc - this name is not real

    if CHAT_MODEL is None and (_busy or _spent):
        CHAT_MODEL = (_busy + _spent)[0]
        print(f'\nno model answered instantly - using {CHAT_MODEL} '
              f'(it exists; requests retry and fail over automatically)', flush=True)

if CHAT_MODEL is None:
    raise RuntimeError(
        'No usable Gemini model found. Check your key at '
        'https://aistudio.google.com/apikey, or set MODEL_OVERRIDE to a model name '
        'shown there.')

# The failover pool: the chosen model first, then every other candidate EXCEPT the
# ones the probe already proved do not exist. Pooling a model that 404s just moves
# the crash later. Anything unprobed that turns out to be retired is struck off by
# _gen() the first time it answers 404.
MODEL_POOL[:] = ([CHAT_MODEL] +
                 [m for m in _ordered if m != CHAT_MODEL and m not in _dead])[:6]

print('\n-> using CHAT_MODEL =', CHAT_MODEL, flush=True)
print('   failover pool  =', MODEL_POOL, flush=True)

# smoke test - goes through _gen, so a 503 retries and a 429 switches model
_t = _gen(model=CHAT_MODEL, contents='Reply with the single word: ready')
print('Gemini says:', (_t.text or '').strip(), flush=True)
print('\nSetup complete. You can run the rest of the notebook.', flush=True)
# ============================================================================
# --- from notebook cell 6 ---
# ============================================================================
# ============================ EDIT ME =====================================
# (brand, generic, common_strength, drug_class, otc_or_rx)
BRANDS = [
    ('Panadol', 'paracetamol', '500 mg', 'analgesic/antipyretic', 'OTC'),
    ('Panadol Extra', 'paracetamol + caffeine', '500/65 mg', 'analgesic', 'OTC'),
    ('Panadol CF', 'paracetamol + pseudoephedrine + chlorpheniramine', 'combo', 'cold remedy', 'OTC'),
    ('Calpol', 'paracetamol', '120 mg/5 mL syrup', 'analgesic/antipyretic', 'OTC'),
    ('Febrol', 'paracetamol', '120 mg/5 mL syrup', 'analgesic/antipyretic', 'OTC'),
    ('Cetal', 'paracetamol', '500 mg', 'analgesic/antipyretic', 'OTC'),
    ('Flu-Out', 'paracetamol + phenylephrine + chlorpheniramine', 'combo', 'cold remedy', 'OTC'),
    ('Arinac', 'ibuprofen + pseudoephedrine', '200/30 mg', 'cold remedy', 'OTC'),
    ('Arinac Forte', 'ibuprofen + pseudoephedrine', '400/60 mg', 'cold remedy', 'OTC'),
    ('Brufen', 'ibuprofen', '400 mg', 'NSAID', 'OTC'),
    ('Ibucin', 'ibuprofen', '400 mg', 'NSAID', 'OTC'),
    ('Ponstan', 'mefenamic acid', '500 mg', 'NSAID', 'Rx'),
    ('Voltaral', 'diclofenac', '50 mg', 'NSAID', 'Rx'),
    ('Dicloran', 'diclofenac', '50 mg', 'NSAID', 'Rx'),
    ('Synflex', 'naproxen', '500 mg', 'NSAID', 'Rx'),
    ('Toradol', 'ketorolac', '10 mg', 'NSAID', 'Rx'),
    ('Nims', 'nimesulide', '100 mg', 'NSAID', 'Rx'),
    ('Disprin', 'aspirin', '300 mg', 'NSAID/antiplatelet', 'OTC'),
    ('Ascard', 'aspirin', '75 mg', 'antiplatelet', 'Rx'),
    ('Loprin', 'aspirin', '75 mg', 'antiplatelet', 'Rx'),
    ('Augmentin', 'amoxicillin + clavulanic acid', '625 mg', 'antibiotic', 'Rx'),
    ('Calamox', 'amoxicillin + clavulanic acid', '625 mg', 'antibiotic', 'Rx'),
    ('Amoxil', 'amoxicillin', '500 mg', 'antibiotic', 'Rx'),
    ('Velosef', 'cefradine', '500 mg', 'antibiotic', 'Rx'),
    ('Ceporex', 'cefalexin', '500 mg', 'antibiotic', 'Rx'),
    ('Cefspan', 'cefixime', '400 mg', 'antibiotic', 'Rx'),
    ('Ciproxin', 'ciprofloxacin', '500 mg', 'antibiotic', 'Rx'),
    ('Novidat', 'ciprofloxacin', '500 mg', 'antibiotic', 'Rx'),
    ('Avelox', 'moxifloxacin', '400 mg', 'antibiotic', 'Rx'),
    ('Klaricid', 'clarithromycin', '500 mg', 'antibiotic', 'Rx'),
    ('Azomax', 'azithromycin', '500 mg', 'antibiotic', 'Rx'),
    ('Zetro', 'azithromycin', '500 mg', 'antibiotic', 'Rx'),
    ('Flagyl', 'metronidazole', '400 mg', 'antibiotic', 'Rx'),
    ('Metrozine', 'metronidazole', '400 mg', 'antibiotic', 'Rx'),
    ('Risek', 'omeprazole', '20 mg', 'PPI', 'Rx'),
    ('Nexum', 'esomeprazole', '40 mg', 'PPI', 'Rx'),
    ('Zantac', 'ranitidine', '150 mg', 'H2 blocker', 'Rx'),
    ('Motilium', 'domperidone', '10 mg', 'antiemetic', 'Rx'),
    ('Buscopan', 'hyoscine butylbromide', '10 mg', 'antispasmodic', 'OTC'),
    ('Imodium', 'loperamide', '2 mg', 'antidiarrhoeal', 'OTC'),
    ('Peditral', 'oral rehydration salts', 'sachet', 'rehydration', 'OTC'),
    ('ORS', 'oral rehydration salts', 'sachet', 'rehydration', 'OTC'),
    ('Glucophage', 'metformin', '500 mg', 'antidiabetic', 'Rx'),
    ('Neodipar', 'metformin', '500 mg', 'antidiabetic', 'Rx'),
    ('Amaryl', 'glimepiride', '2 mg', 'antidiabetic', 'Rx'),
    ('Januvia', 'sitagliptin', '100 mg', 'antidiabetic', 'Rx'),
    ('Tenormin', 'atenolol', '50 mg', 'beta blocker', 'Rx'),
    ('Concor', 'bisoprolol', '5 mg', 'beta blocker', 'Rx'),
    ('Inderal', 'propranolol', '10 mg', 'beta blocker', 'Rx'),
    ('Norvasc', 'amlodipine', '5 mg', 'calcium channel blocker', 'Rx'),
    ('Amlong', 'amlodipine', '5 mg', 'calcium channel blocker', 'Rx'),
    ('Zestril', 'lisinopril', '5 mg', 'ACE inhibitor', 'Rx'),
    ('Capoten', 'captopril', '25 mg', 'ACE inhibitor', 'Rx'),
    ('Coversyl', 'perindopril', '4 mg', 'ACE inhibitor', 'Rx'),
    ('Losar', 'losartan', '50 mg', 'ARB', 'Rx'),
    ('Provas', 'valsartan', '80 mg', 'ARB', 'Rx'),
    ('Lasix', 'furosemide', '40 mg', 'diuretic', 'Rx'),
    ('Aldactone', 'spironolactone', '25 mg', 'diuretic', 'Rx'),
    ('Lipitor', 'atorvastatin', '20 mg', 'statin', 'Rx'),
    ('Zocor', 'simvastatin', '20 mg', 'statin', 'Rx'),
    ('Plavix', 'clopidogrel', '75 mg', 'antiplatelet', 'Rx'),
    ('Coumadin', 'warfarin', '5 mg', 'anticoagulant', 'Rx'),
    ('Warfarin', 'warfarin', '5 mg', 'anticoagulant', 'Rx'),
    ('Xarelto', 'rivaroxaban', '20 mg', 'anticoagulant', 'Rx'),
    ('Ventolin', 'salbutamol', '100 mcg inhaler', 'bronchodilator', 'Rx'),
    ('Seretide', 'fluticasone + salmeterol', 'inhaler', 'asthma controller', 'Rx'),
    ('Deltacortril', 'prednisolone', '5 mg', 'corticosteroid', 'Rx'),
    ('Decadron', 'dexamethasone', '0.5 mg', 'corticosteroid', 'Rx'),
    ('Zyrtec', 'cetirizine', '10 mg', 'antihistamine', 'OTC'),
    ('Softin', 'cetirizine', '10 mg', 'antihistamine', 'OTC'),
    ('Loratin', 'loratadine', '10 mg', 'antihistamine', 'OTC'),
    ('Avil', 'pheniramine', '22.5 mg', 'antihistamine', 'OTC'),
    ('Piriton', 'chlorpheniramine', '4 mg', 'antihistamine', 'OTC'),
    ('Xanax', 'alprazolam', '0.5 mg', 'benzodiazepine', 'Rx'),
    ('Lexotanil', 'bromazepam', '3 mg', 'benzodiazepine', 'Rx'),
    ('Rivotril', 'clonazepam', '0.5 mg', 'benzodiazepine', 'Rx'),
    ('Lexapro', 'escitalopram', '10 mg', 'SSRI', 'Rx'),
    ('Prozac', 'fluoxetine', '20 mg', 'SSRI', 'Rx'),
    ('Tegral', 'carbamazepine', '200 mg', 'anticonvulsant', 'Rx'),
    ('Epival', 'sodium valproate', '500 mg', 'anticonvulsant', 'Rx'),
    ('Tramal', 'tramadol', '50 mg', 'opioid analgesic', 'Rx'),
    ('Thyrox', 'levothyroxine', '50 mcg', 'thyroid hormone', 'Rx'),
    ('Neurobion', 'vitamin B complex', 'tablet', 'supplement', 'OTC'),
    ('CaC-1000', 'calcium + vitamin C', 'effervescent', 'supplement', 'OTC'),
    ('Fefol', 'ferrous sulfate + folic acid', 'capsule', 'supplement', 'OTC'),
    ('Diflucan', 'fluconazole', '150 mg', 'antifungal', 'Rx'),
]

# (generic_a, generic_b, severity, mechanism, advice)   severity: major|moderate|minor
INTERACTIONS = [
    ('warfarin', 'aspirin', 'major', 'Additive antiplatelet and anticoagulant effect', 'Significantly raises bleeding risk. Do not combine unless a cardiologist has specifically prescribed both.'),
    ('warfarin', 'ibuprofen', 'major', 'NSAID inhibits platelets and irritates gastric mucosa', 'High risk of gastrointestinal bleeding. Paracetamol is the safer pain option on warfarin.'),
    ('warfarin', 'diclofenac', 'major', 'NSAID inhibits platelets and irritates gastric mucosa', 'High bleeding risk. Avoid and discuss alternatives with the prescriber.'),
    ('warfarin', 'metronidazole', 'major', 'Inhibits CYP2C9 metabolism of warfarin', 'INR can rise sharply within days. Requires INR monitoring or a different antibiotic.'),
    ('warfarin', 'clarithromycin', 'major', 'CYP3A4 and CYP2C9 inhibition raises warfarin levels', 'Raises bleeding risk. Needs an INR check if unavoidable.'),
    ('warfarin', 'fluconazole', 'major', 'Potent CYP2C9 inhibition', 'Can cause dangerous INR elevation. Avoid or monitor closely.'),
    ('warfarin', 'paracetamol', 'moderate', 'Regular high-dose paracetamol can raise INR', 'Occasional doses are fine. Daily use above 2 g needs INR monitoring.'),
    ('clopidogrel', 'omeprazole', 'moderate', 'CYP2C19 inhibition reduces conversion of clopidogrel to its active form', 'May reduce antiplatelet protection. Pantoprazole is usually preferred.'),
    ('aspirin', 'ibuprofen', 'moderate', 'Ibuprofen competitively blocks aspirin binding to platelets', 'Can cancel out low-dose aspirin heart protection. Take aspirin 2 hours before ibuprofen.'),
    ('ibuprofen', 'diclofenac', 'major', 'Duplicate NSAID therapy', 'Two NSAIDs together sharply raise ulcer, bleeding and kidney risk with no added pain relief.'),
    ('ibuprofen', 'naproxen', 'major', 'Duplicate NSAID therapy', 'Do not combine two NSAIDs.'),
    ('ibuprofen', 'mefenamic acid', 'major', 'Duplicate NSAID therapy', 'Do not combine two NSAIDs.'),
    ('ibuprofen', 'prednisolone', 'major', 'Additive gastric mucosal damage', 'Marked increase in peptic ulcer and GI bleed risk. Needs gastric protection if unavoidable.'),
    ('ibuprofen', 'lisinopril', 'major', 'NSAID plus ACE inhibitor reduces renal perfusion', "Part of the 'triple whammy'. Raises blood pressure and risks acute kidney injury, especially alongside a diuretic."),
    ('ibuprofen', 'furosemide', 'major', 'NSAID blunts the diuretic and reduces renal perfusion', 'Risk of acute kidney injury and worsening fluid overload.'),
    ('ibuprofen', 'losartan', 'moderate', 'Reduced renal perfusion and blunted blood pressure control', 'Monitor kidney function and blood pressure.'),
    ('diclofenac', 'lisinopril', 'major', 'NSAID plus ACE inhibitor reduces renal perfusion', 'Risk of acute kidney injury. Avoid the NSAID if possible.'),
    ('lisinopril', 'spironolactone', 'major', 'Both raise serum potassium', 'Risk of dangerous hyperkalaemia. Needs potassium blood monitoring.'),
    ('losartan', 'spironolactone', 'major', 'Both raise serum potassium', 'Risk of hyperkalaemia. Requires monitoring.'),
    ('tramadol', 'escitalopram', 'major', 'Additive serotonergic activity', 'Risk of serotonin syndrome and seizures. Avoid the combination.'),
    ('tramadol', 'fluoxetine', 'major', 'Additive serotonergic activity plus CYP2D6 inhibition', 'Serotonin syndrome and seizure risk.'),
    ('tramadol', 'alprazolam', 'major', 'Additive central nervous system and respiratory depression', 'Risk of over-sedation and respiratory depression. Avoid.'),
    ('tramadol', 'clonazepam', 'major', 'Additive central nervous system and respiratory depression', 'Avoid combining opioids with benzodiazepines.'),
    ('escitalopram', 'ibuprofen', 'moderate', 'SSRIs impair platelet serotonin while NSAIDs damage mucosa', 'Increased gastrointestinal bleeding risk. Consider gastric protection.'),
    ('fluoxetine', 'aspirin', 'moderate', 'SSRI plus antiplatelet', 'Increased gastrointestinal bleeding risk.'),
    ('clarithromycin', 'atorvastatin', 'major', 'CYP3A4 inhibition raises statin levels', 'Risk of muscle damage (rhabdomyolysis). The statin is usually paused during the antibiotic course.'),
    ('clarithromycin', 'simvastatin', 'major', 'Strong CYP3A4 inhibition', 'Contraindicated combination. Risk of rhabdomyolysis.'),
    ('clarithromycin', 'domperidone', 'major', 'Both prolong the QT interval', 'Risk of serious heart rhythm disturbance.'),
    ('azithromycin', 'domperidone', 'moderate', 'Additive QT prolongation', 'Use with caution, especially with low potassium.'),
    ('ciprofloxacin', 'calcium + vitamin C', 'moderate', 'Divalent cations chelate the fluoroquinolone', 'Calcium blocks absorption. Separate the doses by at least 2 hours.'),
    ('ciprofloxacin', 'ferrous sulfate + folic acid', 'moderate', 'Iron chelates the fluoroquinolone', 'Reduces antibiotic absorption. Separate doses by 2 hours.'),
    ('ciprofloxacin', 'domperidone', 'moderate', 'Additive QT prolongation', 'Caution advised.'),
    ('levothyroxine', 'calcium + vitamin C', 'moderate', 'Calcium binds levothyroxine in the gut', 'Take levothyroxine on an empty stomach, 4 hours apart from calcium.'),
    ('levothyroxine', 'ferrous sulfate + folic acid', 'moderate', 'Iron binds levothyroxine', 'Separate doses by at least 4 hours.'),
    ('levothyroxine', 'omeprazole', 'moderate', 'Reduced gastric acid lowers absorption', 'Thyroid levels may need rechecking.'),
    ('metronidazole', 'alcohol', 'major', 'Disulfiram-like reaction', 'Causes severe flushing, vomiting and palpitations. Avoid alcohol during and for 48 hours after the course.'),
    ('carbamazepine', 'sodium valproate', 'moderate', 'Complex mutual metabolic interference', 'Blood level monitoring required. Specialist decision only.'),
    ('propranolol', 'salbutamol', 'major', 'Beta blockade opposes bronchodilation', 'Can trigger bronchospasm in asthma. Non-selective beta blockers are generally avoided in asthma.'),
    ('atenolol', 'salbutamol', 'moderate', 'Partial opposition of the bronchodilator effect', 'Use cardioselective beta blockers cautiously in asthma.'),
    ('metformin', 'furosemide', 'moderate', 'Dehydration raises lactic acidosis risk', 'Maintain hydration. Caution during acute illness.'),
    ('prednisolone', 'metformin', 'moderate', 'Steroids raise blood glucose', 'Blood sugar control may worsen. Monitor more often.'),
    ('amlodipine', 'simvastatin', 'moderate', 'Amlodipine raises simvastatin exposure', 'Simvastatin dose is usually capped at 20 mg.'),
    ('alprazolam', 'clarithromycin', 'moderate', 'CYP3A4 inhibition raises benzodiazepine levels', 'Increased sedation risk.'),
    ('furosemide', 'spironolactone', 'minor', 'Opposing effects on potassium, commonly co-prescribed', 'Usually intentional. Potassium should still be monitored.'),
]

# (generic, active_ingredients)  - combination products, '|' separated.
# This is what catches "Panadol + Panadol Extra + Flu-Out = 3x paracetamol".
INGREDIENT_MAP = [
    ('paracetamol', 'paracetamol'),
    ('paracetamol + caffeine', 'paracetamol|caffeine'),
    ('paracetamol + pseudoephedrine + chlorpheniramine', 'paracetamol|pseudoephedrine|chlorpheniramine'),
    ('paracetamol + phenylephrine + chlorpheniramine', 'paracetamol|phenylephrine|chlorpheniramine'),
    ('ibuprofen + pseudoephedrine', 'ibuprofen|pseudoephedrine'),
    ('amoxicillin + clavulanic acid', 'amoxicillin|clavulanic acid'),
    ('fluticasone + salmeterol', 'fluticasone|salmeterol'),
    ('calcium + vitamin C', 'calcium|vitamin c'),
    ('ferrous sulfate + folic acid', 'ferrous sulfate|folic acid'),
]

# (generic, unit, max_single_dose, max_daily_dose, note)   max_daily 0 = clinician-directed
MAX_DOSES = [
    ('paracetamol', 'mg', 1000, 4000, 'Reduce to 3000 mg/day with liver disease, alcohol use or body weight under 50 kg. Overdose causes irreversible liver failure.'),
    ('ibuprofen', 'mg', 800, 2400, 'Self-medication limit is 1200 mg/day. Take with food.'),
    ('diclofenac', 'mg', 75, 150, 'Cardiovascular risk rises above 100 mg/day.'),
    ('naproxen', 'mg', 500, 1000, 'Take with food.'),
    ('mefenamic acid', 'mg', 500, 1500, 'Not for use beyond 7 days.'),
    ('aspirin', 'mg', 1000, 4000, "Antiplatelet dose is only 75-150 mg/day. Never give to children under 16 with a viral illness (Reye's syndrome)."),
    ('amoxicillin', 'mg', 1000, 3000, 'Complete the full prescribed course.'),
    ('amoxicillin + clavulanic acid', 'mg', 1000, 2625, 'The clavulanate component should not exceed 375 mg/day.'),
    ('azithromycin', 'mg', 500, 500, 'Typically 500 mg once daily for 3 days.'),
    ('ciprofloxacin', 'mg', 750, 1500, 'Avoid in under-18s and in pregnancy unless specialist-directed. Tendon rupture risk.'),
    ('metronidazole', 'mg', 500, 2000, 'No alcohol during or for 48 hours after the course.'),
    ('cefixime', 'mg', 400, 400, 'Usually 200 mg twice daily.'),
    ('omeprazole', 'mg', 40, 40, 'Long-term use is linked to B12 and magnesium deficiency.'),
    ('metformin', 'mg', 1000, 2000, 'Take with meals. Stop and seek care during severe dehydration or vomiting.'),
    ('atorvastatin', 'mg', 80, 80, 'Report unexplained muscle pain immediately.'),
    ('amlodipine', 'mg', 10, 10, 'Ankle swelling is the most common side effect.'),
    ('losartan', 'mg', 100, 100, 'Contraindicated in pregnancy.'),
    ('lisinopril', 'mg', 40, 40, 'Contraindicated in pregnancy. Dry cough is common.'),
    ('cetirizine', 'mg', 10, 10, 'Can cause drowsiness.'),
    ('loratadine', 'mg', 10, 10, 'Less sedating than cetirizine.'),
    ('tramadol', 'mg', 100, 400, 'Dependence risk. Lower the ceiling in elderly patients.'),
    ('levothyroxine', 'mcg', 200, 200, 'Dose is individualised by TSH blood tests. Never self-adjust.'),
    ('loperamide', 'mg', 4, 16, 'Do not use if there is fever or blood in the stool.'),
    ('prednisolone', 'mg', 0, 0, 'Dose is entirely clinician-directed. Never stop long courses abruptly.'),
]

# (pattern, category, urgency, message_en, message_ur)   patterns are '|' separated substrings
RED_FLAGS = [
    ('chest pain|pressure in chest|pain in left arm|seene mein dard', 'cardiac', 'emergency', 'Chest pain or pressure, especially spreading to the arm, jaw or back, can be a heart attack.', 'Seene mein dard ya dabao dil ke daure ki nishani ho sakta hai.'),
    ('difficulty breathing|shortness of breath|cannot breathe|saans', 'respiratory', 'emergency', 'Difficulty breathing needs immediate assessment.', 'Saans lene mein takleef par foran doctor se rabta karein.'),
    ('weakness on one side|face drooping|slurred speech|cannot speak', 'stroke', 'emergency', 'One-sided weakness, facial drooping or slurred speech are stroke signs. Time is critical.', 'Jism ke aik taraf kamzori ya zubaan lagharzana falij ki alamat hai.'),
    ('unconscious|fainted|not responding|seizure|fit', 'neurological', 'emergency', 'Loss of consciousness or a seizure requires emergency care.', 'Behoshi ya doura parne par foran hospital jayein.'),
    ('blood in vomit|vomiting blood|black stool|tarry stool|blood in stool', 'gi bleed', 'emergency', 'Vomiting blood or black tarry stools indicate internal bleeding.', 'Khoon ki ulti ya kaala pakhana andaruni khoon behne ki alamat hai.'),
    ('coughing blood|blood in sputum', 'respiratory', 'urgent', 'Coughing up blood requires prompt medical assessment.', 'Khansi mein khoon aana foran muaina chahta hai.'),
    ('severe abdominal pain|rigid abdomen|pait mein shadeed dard', 'abdominal', 'urgent', 'Severe or worsening abdominal pain needs same-day examination.', 'Pait mein shadeed dard par usi din doctor ko dikhayein.'),
    ('stiff neck|neck stiffness with fever|light hurts eyes', 'meningitis', 'emergency', 'Fever with neck stiffness or light sensitivity may be meningitis.', 'Bukhar ke sath gardan akarna meningitis ho sakta hai.'),
    ('swelling of face|swollen lips|swollen tongue|throat closing|rash all over', 'anaphylaxis', 'emergency', 'Facial or tongue swelling with a spreading rash suggests a severe allergic reaction.', 'Chehre ya zubaan ka soojna shadeed allergy ki alamat hai.'),
    ('baby under 3 months fever|newborn fever|infant fever', 'paediatric', 'emergency', 'Any fever in an infant under 3 months is an emergency.', 'Teen maheene se chhote bachay ko bukhar ho to foran hospital le jayein.'),
    ('not passing urine|no urine|sunken eyes|very dry mouth|severe dehydration', 'dehydration', 'urgent', 'Signs of severe dehydration need urgent rehydration and assessment.', 'Peshab band hona ya aankhein dhansna shadeed pani ki kami hai.'),
    ('suicidal|want to die|end my life|kill myself|harm myself', 'mental health', 'emergency', 'Thoughts of self-harm deserve immediate support from a crisis service or a trusted person.', 'Khud ko nuqsan pohanchane ke khayalat par foran madad lein.'),
    ('overdose|took too many tablets|swallowed pills', 'poisoning', 'emergency', 'A suspected overdose is a medical emergency even if the person feels fine.', 'Zyada goliyan khane par foran hospital jayein, chahe tabiyat theek lage.'),
    ('bleeding that will not stop|heavy bleeding', 'haemorrhage', 'emergency', 'Uncontrolled bleeding requires emergency care.', 'Khoon band na ho to foran hospital jayein.'),
    ('yellow eyes|yellow skin|jaundice', 'hepatic', 'urgent', 'Yellowing of the eyes or skin suggests liver problems and needs testing.', 'Aankhon ya jild ka peela hona jigar ki kharabi ki alamat hai.'),
    ('vision loss|sudden blurred vision|cannot see', 'ophthalmic', 'emergency', 'Sudden vision change needs same-day specialist assessment.', 'Achanak nazar ka chala jana foran muaina chahta hai.'),
    ('severe headache worst ever|thunderclap headache', 'neurological', 'emergency', "A sudden, severe 'worst ever' headache needs emergency imaging.", 'Achanak shadeed sar dard par foran hospital jayein.'),
    ('pregnant|pregnancy|breastfeeding|expecting', 'pregnancy', 'caution', 'Many medicines are unsafe in pregnancy or breastfeeding. A clinician must review every drug.', 'Hamal ya doodh pilane ke doran har dawa doctor se check karwayein.'),
]
# ========================== END EDIT ME ===================================

brands_df = pd.DataFrame(BRANDS, columns=['brand', 'generic', 'common_strength', 'drug_class', 'otc_or_rx'])
inter_df  = pd.DataFrame(INTERACTIONS, columns=['generic_a', 'generic_b', 'severity', 'mechanism', 'advice'])
ingred_df = pd.DataFrame(INGREDIENT_MAP, columns=['generic', 'active_ingredients'])
dose_df   = pd.DataFrame(MAX_DOSES, columns=['generic', 'unit', 'max_single_dose', 'max_daily_dose', 'note'])
flags_df  = pd.DataFrame(RED_FLAGS, columns=['pattern', 'category', 'urgency', 'message_en', 'message_ur'])

BRAND2GENERIC = {r.brand.lower(): r.generic for r in brands_df.itertuples()}
BRAND_META    = {r.brand.lower(): r for r in brands_df.itertuples()}
ALL_GENERICS  = sorted(set(brands_df.generic))
INGREDIENTS   = {r.generic: r.active_ingredients.split('|') for r in ingred_df.itertuples()}
MAX_DOSE      = {r.generic: r for r in dose_df.itertuples()}

print(f'{len(brands_df)} brands | {len(inter_df)} interactions | '
      f'{len(dose_df)} dose limits | {len(flags_df)} red flags')
brands_df.sample(5, random_state=1)
# ============================================================================
# --- from notebook cell 7 ---
# ============================================================================
# Offline drug summaries. Used if openFDA is unreachable on demo day, so the
# whole notebook still runs with the wifi unplugged.
FALLBACK_TEXT = {
    'paracetamol':
        'Paracetamol (acetaminophen) treats mild to moderate pain and reduces fever. It does not reduce inflammation. The usual adult dose is 500-1000 mg every 4-6 hours, with a hard ceiling of 4000 mg in 24 hours; 3000 mg is safer for people with liver disease, low body weight or regular alcohol use. It is the preferred painkiller in pregnancy, in asthma, in peptic ulcer disease and for people taking blood thinners. The main danger is overdose: exceeding the daily limit, often by unknowingly combining several branded cold-and-flu products that each contain paracetamol, can cause irreversible liver failure with few early symptoms. Seek emergency care after any suspected overdose even if the person feels well.',
    'ibuprofen':
        'Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) used for pain, fever and inflammatory conditions. The typical adult dose is 200-400 mg every 6-8 hours; self-medication should stay at or below 1200 mg per day and prescribed use below 2400 mg. Always take with or after food. It can cause stomach irritation, ulcers and bleeding, can raise blood pressure, and can reduce kidney function, particularly in dehydration, in the elderly, or when combined with an ACE inhibitor and a diuretic. Avoid in the third trimester of pregnancy, in active peptic ulcer disease, in severe heart failure and in people with aspirin-sensitive asthma. Never combine with another NSAID.',
    'diclofenac':
        'Diclofenac is a potent NSAID used for musculoskeletal and dental pain. The usual adult dose is 50 mg two to three times daily, maximum 150 mg per day, taken with food. It carries a higher cardiovascular risk than ibuprofen and is avoided in established heart disease, stroke and peripheral arterial disease. Gastrointestinal bleeding, kidney impairment and raised liver enzymes are recognised risks. Topical gel is a lower-risk option for localised joint pain.',
    'aspirin':
        "Aspirin at 75-150 mg daily is used long-term to prevent heart attack and stroke by inhibiting platelets. At 300-1000 mg it acts as a painkiller and antipyretic, up to 4000 mg per day. It must not be given to children or teenagers under 16 with a viral illness because of Reye's syndrome. Bleeding risk, gastric irritation and ulceration are the principal harms, greatly increased when combined with anticoagulants, other NSAIDs, corticosteroids or SSRIs. Do not stop a cardiology-prescribed aspirin without medical advice.",
    'amoxicillin + clavulanic acid':
        'Co-amoxiclav combines amoxicillin with clavulanic acid to overcome bacterial resistance. It treats respiratory, urinary, dental, skin and abdominal infections. A common adult regimen is 625 mg every 8 hours or 1 g every 12 hours for 5-7 days. Diarrhoea is the commonest side effect; it can also cause nausea, rash and, rarely, cholestatic liver injury. It must be avoided in penicillin allergy. The full course should be completed. It is not effective against viral illness such as the common cold or most sore throats, and unnecessary use drives antibiotic resistance.',
    'azithromycin':
        'Azithromycin is a macrolide antibiotic used for respiratory tract, skin and some sexually transmitted infections, and as an alternative in penicillin allergy. A typical adult course is 500 mg once daily for three days. It can prolong the QT interval on the ECG, so caution is needed alongside other QT-prolonging drugs and in low potassium or magnesium. Nausea, abdominal discomfort and diarrhoea are common.',
    'ciprofloxacin':
        'Ciprofloxacin is a fluoroquinolone antibiotic used for urinary, gastrointestinal and some respiratory infections. Adult doses are typically 250-750 mg twice daily. Because of a risk of tendon rupture, nerve damage and central nervous system effects, it is reserved for infections where other antibiotics are unsuitable, and it is generally avoided in children, adolescents, pregnancy and breastfeeding. Absorption is markedly reduced by calcium, iron, zinc, magnesium and antacids, which must be separated by at least two hours. It can prolong the QT interval and can raise the effect of theophylline and tizanidine.',
    'metronidazole':
        'Metronidazole treats anaerobic bacterial infections and protozoal infections including amoebiasis and giardiasis, and is common in dental and abdominal infections. The adult dose is often 400 mg three times daily for 5-7 days. A metallic taste, nausea and dark urine are common. Alcohol must be avoided during treatment and for 48 hours afterwards because of a disulfiram-like reaction causing flushing, vomiting and palpitations. It significantly increases the effect of warfarin.',
    'omeprazole':
        'Omeprazole is a proton pump inhibitor that reduces stomach acid, used for reflux, gastritis, peptic ulcer and to protect the stomach during NSAID treatment. The usual adult dose is 20 mg once daily before breakfast, up to 40 mg. It should be reviewed rather than continued indefinitely: prolonged use is associated with vitamin B12 deficiency, low magnesium, and a small increase in fracture and enteric infection risk. It reduces the effectiveness of clopidogrel and lowers the absorption of levothyroxine and some antifungals.',
    'metformin':
        'Metformin is first-line treatment for type 2 diabetes. It lowers blood glucose by reducing hepatic glucose production and improving insulin sensitivity, and does not itself cause hypoglycaemia. Treatment starts at 500 mg once or twice daily with meals and is increased gradually to a usual maximum of 2000 mg per day. Nausea, diarrhoea and a metallic taste are common and usually settle. It must be paused during severe dehydration, vomiting, serious infection, or before contrast imaging, because of a rare but serious risk of lactic acidosis. Kidney function should be checked periodically.',
    'amlodipine':
        'Amlodipine is a calcium channel blocker used for high blood pressure and angina. The usual dose is 5-10 mg once daily. Ankle swelling, flushing, headache and palpitations are the common side effects; ankle swelling is dose-related and is not relieved by diuretics. It is generally safe in asthma and diabetes. Blood pressure medicines should not be stopped abruptly without advice.',
    'atorvastatin':
        'Atorvastatin lowers LDL cholesterol and reduces cardiovascular events. Doses range from 10 to 80 mg once daily, taken at any time of day. Muscle aches are the most reported complaint; unexplained, severe or widespread muscle pain with dark urine must be reported immediately as it can indicate rhabdomyolysis. Risk rises sharply when combined with clarithromycin, some antifungals, or large quantities of grapefruit juice. Liver enzymes are checked before and during treatment. It is contraindicated in pregnancy.',
    'warfarin':
        'Warfarin is an anticoagulant used to prevent and treat clots in atrial fibrillation, venous thromboembolism and mechanical heart valves. The dose is individual and guided by the INR blood test, usually targeting 2.0-3.0. Its effect is altered by a very large number of drugs, by illness, and by dietary vitamin K from green leafy vegetables, which should be kept consistent rather than avoided. Antibiotics, antifungals, NSAIDs, aspirin and regular high-dose paracetamol all increase risk. Any unusual bruising, nosebleeds, blood in urine or stool, or black stools must be reported immediately.',
    'salbutamol':
        'Salbutamol is a short-acting bronchodilator inhaler that relieves asthma symptoms within minutes. The usual reliever dose is 100-200 micrograms as needed. Needing it more than twice a week, or a reduced response to it, signals poorly controlled asthma and requires review and usually a preventer inhaler. Tremor, palpitations and headache are expected effects. Inhaler technique and, for many patients, a spacer device, matter more than the dose.',
    'cetirizine':
        'Cetirizine is a second-generation antihistamine for allergic rhinitis, urticaria and itching. The adult dose is 10 mg once daily. It is less sedating than older antihistamines such as chlorpheniramine or pheniramine, but still causes drowsiness in some people, so caution is needed when driving. The dose is reduced in kidney impairment.',
    'prednisolone':
        'Prednisolone is an oral corticosteroid used for inflammatory and autoimmune conditions and for asthma exacerbations. Dosing is entirely clinician-directed and varies widely. Short courses can raise blood glucose, blood pressure and mood, and disturb sleep. Courses longer than about three weeks must be tapered rather than stopped abruptly because of adrenal suppression. Long-term use risks osteoporosis, cataracts, infection and gastric ulceration, especially alongside NSAIDs.',
    'levothyroxine':
        'Levothyroxine replaces thyroid hormone in hypothyroidism. It is taken once daily on an empty stomach, at least 30 minutes before food, and the dose is adjusted only on the basis of TSH blood tests, usually rechecked 6-8 weeks after any change. Calcium, iron, antacids and proton pump inhibitors reduce absorption and should be separated by four hours. Over-replacement causes palpitations, tremor, weight loss and, over time, bone loss and atrial fibrillation.',
    'tramadol':
        'Tramadol is a weak opioid analgesic for moderate pain not controlled by simpler drugs. The adult dose is typically 50-100 mg every 4-6 hours to a maximum of 400 mg per day, with lower ceilings in the elderly and in kidney or liver impairment. It lowers the seizure threshold and carries a risk of serotonin syndrome with SSRIs, SNRIs and other serotonergic drugs. Combining it with benzodiazepines or alcohol risks fatal respiratory depression. It causes dependence and should be used for the shortest effective period.',
    'oral rehydration salts':
        'Oral rehydration salts are the primary treatment for dehydration from diarrhoea and are one of the highest-impact interventions in child health. One sachet is dissolved in the exact volume of clean water stated on the packet; using less water makes the solution dangerously concentrated. Small, frequent sips are given, continuing after every loose stool. Breastfeeding and normal feeding should continue. Zinc supplementation for 10-14 days shortens diarrhoea in children. Antibiotics and anti-motility drugs are not routine. Sunken eyes, absent urine, lethargy or blood in the stool require urgent medical care.',
    'loperamide':
        'Loperamide slows gut motility to reduce stool frequency in uncomplicated acute diarrhoea. The adult dose is 4 mg initially then 2 mg after each loose stool, to a maximum of 16 mg per day. It must not be used when there is fever, blood or mucus in the stool, or suspected dysentery, because retaining the infection can cause serious complications. It is not recommended in young children. It does not treat the cause of diarrhoea; fluid replacement does.',
}

FALLBACK_DOCS = {
    g: {'doc_id': 'fallback::' + g.replace(' ', '_'), 'generic': g,
        'title': g.title() + ' - medicine information',
        'source': 'MedSaathi curated summary (offline fallback)', 'text': t}
    for g, t in FALLBACK_TEXT.items()
}
print(len(FALLBACK_DOCS), 'offline fallback documents ready')
# ============================================================================
# --- from notebook cell 9 ---
# ============================================================================
US_NAME = {
    'paracetamol': 'acetaminophen',
    'salbutamol': 'albuterol',
    'adrenaline': 'epinephrine',
    'amoxicillin + clavulanic acid': 'amoxicillin and clavulanate potassium',
    'oral rehydration salts': None,          # not an FDA-labelled product
    'calcium + vitamin C': None,
    'ferrous sulfate + folic acid': None,
    'vitamin B complex': None,
}

LABEL_SECTIONS = [
    ('indications_and_usage', 'Indications and usage'),
    ('dosage_and_administration', 'Dosage and administration'),
    ('warnings_and_cautions', 'Warnings and cautions'),
    ('warnings', 'Warnings'),
    ('drug_interactions', 'Drug interactions'),
    ('contraindications', 'Contraindications'),
    ('adverse_reactions', 'Adverse reactions'),
]

def fetch_openfda(generic, timeout=8):
    # Returns [{section, text}] for one generic, or [] on any failure.
    q = US_NAME.get(generic, generic)
    if q is None:
        return []
    try:
        r = requests.get('https://api.fda.gov/drug/label.json',
                         params={'search': f'openfda.generic_name:"{q}"', 'limit': 1},
                         timeout=timeout)
        if r.status_code != 200:
            return []
        results = r.json().get('results', [])
        if not results:
            return []
        rec, out = results[0], []
        for key, label in LABEL_SECTIONS:
            if rec.get(key):
                text = ' '.join(rec[key]) if isinstance(rec[key], list) else str(rec[key])
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 60:
                    out.append({'section': label, 'text': text})
        return out
    except Exception:
        return []


CORPUS_CACHE = f'{CACHE_DIR}/corpus.json'

def build_corpus(generics, use_cache=True):
    if use_cache and os.path.exists(CORPUS_CACHE):
        with open(CORPUS_CACHE, encoding='utf-8') as f:
            docs = json.load(f)
        print(f'loaded {len(docs)} documents from cache')
        return docs

    docs, live, offline = [], 0, 0
    consecutive_failures, api_up = 0, True
    for i, g in enumerate(generics, 1):
        # Circuit breaker: if openFDA is unreachable, stop hammering it and run
        # entirely offline. Fails in seconds, not minutes - this saves the demo.
        sections = fetch_openfda(g) if api_up else []
        if not sections and US_NAME.get(g, g) is not None:
            consecutive_failures += 1
            if api_up and consecutive_failures >= 4:
                api_up = False
                print('  !! openFDA unreachable after 4 attempts - switching to offline corpus')
        else:
            consecutive_failures = 0

        if sections:
            live += 1
            for j, s in enumerate(sections):
                docs.append({'doc_id': f'openfda::{g}::{j}', 'generic': g,
                             'title': f'{g.title()} - {s["section"]}',
                             'source': 'openFDA drug label (api.fda.gov)',
                             'text': s['text'][:6000]})
        elif g in FALLBACK_DOCS:
            offline += 1
            docs.append(FALLBACK_DOCS[g])
        if i % 10 == 0:
            print(f'  ...{i}/{len(generics)} generics processed')
        if api_up:
            time.sleep(0.2)          # be polite to the public API

    # always include every curated fallback doc too - cheap, and improves recall
    have = {d['doc_id'] for d in docs}
    for d in FALLBACK_DOCS.values():
        if d['doc_id'] not in have:
            docs.append(d)

    with open(CORPUS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f'built {len(docs)} documents  ({live} live from openFDA, {offline} from offline fallback)')
    return docs


TARGET_GENERICS = sorted(set(ALL_GENERICS) | set(MAX_DOSE.keys()))
corpus = build_corpus(TARGET_GENERICS)
print('corpus size:', len(corpus), 'documents')
# ============================================================================
# --- from notebook cell 11 ---
# ============================================================================
def chunk_text(text, size=900, overlap=150):
    words, chunks, cur, cur_len = text.split(), [], [], 0
    for w in words:
        cur.append(w); cur_len += len(w) + 1
        if cur_len >= size:
            chunks.append(' '.join(cur))
            back, keep = 0, []
            for w2 in reversed(cur):                 # build the overlap tail
                back += len(w2) + 1
                keep.insert(0, w2)
                if back >= overlap:
                    break
            cur, cur_len = keep, back
    if cur:
        chunks.append(' '.join(cur))
    return [c for c in chunks if len(c) > 80]


CHUNKS = []
for d in corpus:
    for i, c in enumerate(chunk_text(d['text'])):
        CHUNKS.append({'chunk_id': f'{d["doc_id"]}#{i}', 'generic': d['generic'],
                       'title': d['title'], 'source': d['source'], 'text': c})

print(f'{len(corpus)} documents -> {len(CHUNKS)} chunks')
print('avg chunk length:', int(np.mean([len(c['text']) for c in CHUNKS])), 'chars')
print('\nexample chunk:\n', textwrap.fill(CHUNKS[0]['text'][:400], 100))
# ============================================================================
# --- from notebook cell 13 ---
# ============================================================================
# ---------------------------------------------------------------------------
# Embeddings run LOCALLY by default. This is a deliberate engineering decision,
# not a workaround: embedding ~1200 chunks needs ~80 API calls, and the free
# Gemini tier rate-limits that hard - a terrible thing to discover on demo day.
# MiniLM runs on the Colab CPU in seconds, needs no quota, costs nothing, works
# offline, and is easily accurate enough to retrieve drug-label passages.
# The LLM is reserved for what only an LLM can do: planning, reading photos,
# and explaining. Set EMBED_BACKEND = 'gemini' if you want to compare them.
# ---------------------------------------------------------------------------
EMBED_BACKEND     = 'local'              # 'local' (recommended) or 'gemini'
LOCAL_EMBED_MODEL = 'all-MiniLM-L6-v2'   # 384-dim, ~90 MB, downloads once

_ST = None
def _local_embed(texts):
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        print(f'loading {LOCAL_EMBED_MODEL} (one-off ~90 MB download)...', flush=True)
        _ST = SentenceTransformer(LOCAL_EMBED_MODEL)
        print('   loaded. (Any "unauthenticated requests to the HF Hub" warning '
              'above is harmless.)', flush=True)
    return np.array(_ST.encode(texts, batch_size=64, show_progress_bar=False),
                    dtype='float32')

EMBED_CANDIDATES = list(dict.fromkeys(
    EMBED_DISCOVERED + ['gemini-embedding-001', 'text-embedding-004']))
EMBED_MODEL = None

def _gemini_embed(texts, task_type):
    global EMBED_MODEL
    last_err = None
    for m in ([EMBED_MODEL] if EMBED_MODEL else EMBED_CANDIDATES):
        try:
            out = []
            for i in range(0, len(texts), 16):
                r = _embed_call(
                    model=m, contents=texts[i:i + 16],
                    config=types.EmbedContentConfig(task_type=task_type))
                out.extend([e.values for e in r.embeddings])
            EMBED_MODEL = m
            return np.array(out, dtype='float32')
        except Exception as e:
            last_err = e
            print(f'   {m} unavailable: {str(e)[:80]}', flush=True)
    raise RuntimeError(f'all Gemini embedding models failed: {last_err}')


def embed(texts, task_type='RETRIEVAL_DOCUMENT'):
    global EMBED_BACKEND
    if EMBED_BACKEND == 'gemini':
        try:
            return _gemini_embed(texts, task_type)
        except Exception as e:
            print('Gemini embeddings unavailable -> switching to local.', e, flush=True)
            EMBED_BACKEND = 'local'
    return _local_embed(texts)


# Cache is per-backend: the two models produce different dimensions, so a stale
# cache from the other backend must never be reused.
VEC_CACHE = f'{CACHE_DIR}/vectors_{EMBED_BACKEND}.npy'
if os.path.exists(VEC_CACHE) and len(np.load(VEC_CACHE)) == len(CHUNKS):
    MATRIX = np.load(VEC_CACHE)
    print(f'loaded cached vectors {MATRIX.shape} from {VEC_CACHE}', flush=True)
else:
    _t0 = time.time()
    print(f'embedding {len(CHUNKS)} chunks with the {EMBED_BACKEND} backend...', flush=True)
    MATRIX = embed([c['text'] for c in CHUNKS], 'RETRIEVAL_DOCUMENT')
    np.save(f'{CACHE_DIR}/vectors_{EMBED_BACKEND}.npy', MATRIX)
    print(f'embedded {MATRIX.shape} in {time.time()-_t0:.0f}s', flush=True)

MATRIX = MATRIX / (np.linalg.norm(MATRIX, axis=1, keepdims=True) + 1e-9)
print('vector store ready:', MATRIX.shape[0], 'chunks x', MATRIX.shape[1], 'dims')
# ============================================================================
# --- from notebook cell 14 ---
# ============================================================================
STOP = set('a an the of and or to in for with on is are be can do does how what when'.split())

def _tokens(s):
    return {t for t in re.findall(r'[a-z]+', s.lower()) if len(t) > 2 and t not in STOP}

CHUNK_TOKENS = [_tokens(c['text'] + ' ' + c['generic']) for c in CHUNKS]


_QCACHE = {}          # query -> unit vector; the eval re-asks the same questions

def retrieve(query, k=4, alpha=0.65):
    # Hybrid dense + lexical retrieval. alpha=1.0 is dense-only, 0.0 is lexical-only.
    if query not in _QCACHE:
        qv = embed([query], 'RETRIEVAL_QUERY')[0]
        _QCACHE[query] = qv / (np.linalg.norm(qv) + 1e-9)
    qv = _QCACHE[query]
    dense = MATRIX @ qv

    qt = _tokens(query)
    lex = (np.array([len(qt & ct) / len(qt) for ct in CHUNK_TOKENS], dtype='float32')
           if qt else np.zeros(len(CHUNKS), dtype='float32'))

    score = alpha * dense + (1 - alpha) * lex
    idx = np.argsort(-score)[:k]
    return [{**CHUNKS[i], 'score': float(score[i]),
             'dense': float(dense[i]), 'lexical': float(lex[i])} for i in idx]


for r in retrieve('can I drink alcohol while taking metronidazole?', k=3):
    print(f"[{r['score']:.3f}]  {r['generic']:<18} {r['title'][:55]}")
    print('   ', textwrap.fill(r['text'][:220], 100, subsequent_indent='    '), '\n')
# ============================================================================
# --- from notebook cell 16 ---
# ============================================================================
# ---------------------------------------------------------------- TOOL 1
def resolve_medicine(name: str) -> dict:
    # Map a Pakistani brand name (or a generic, or a typo) to its generic ingredient.
    q = re.sub(r'\s+', ' ', str(name).strip().lower())
    q = re.sub(r'\b(\d+\s*(mg|mcg|ml|g)|tablet|tab|cap|capsule|syrup|inj)\b', '', q)
    q = re.sub(r'\s+', ' ', q).strip()

    hit, how = None, None
    if q in BRAND2GENERIC:
        hit, how = q, 'exact brand'
    elif q in {g.lower() for g in ALL_GENERICS}:
        gen = next(g for g in ALL_GENERICS if g.lower() == q)
        return {'input': name, 'generic': gen, 'matched_as': 'exact generic',
                'active_ingredients': INGREDIENTS.get(gen, [gen]), 'confidence': 'high'}
    else:
        close = get_close_matches(q, list(BRAND2GENERIC), n=1, cutoff=0.75)
        if close:
            hit, how = close[0], 'fuzzy brand match'
        else:
            close_g = get_close_matches(q, [g.lower() for g in ALL_GENERICS], n=1, cutoff=0.75)
            if close_g:
                gen = next(g for g in ALL_GENERICS if g.lower() == close_g[0])
                return {'input': name, 'generic': gen, 'matched_as': 'fuzzy generic match',
                        'active_ingredients': INGREDIENTS.get(gen, [gen]), 'confidence': 'medium'}

    if not hit:
        return {'input': name, 'generic': None, 'matched_as': 'not found', 'confidence': 'none',
                'note': 'Not in the local Pakistani brand database. Ask the user for the generic '
                        'name printed on the strip.'}

    m = BRAND_META[hit]
    gen = m.generic
    return {'input': name, 'brand': m.brand, 'generic': gen, 'matched_as': how,
            'common_strength': m.common_strength, 'drug_class': m.drug_class,
            'prescription_status': m.otc_or_rx,
            'active_ingredients': INGREDIENTS.get(gen, [gen]),
            'confidence': 'high' if how == 'exact brand' else 'medium'}


# ---------------------------------------------------------------- TOOL 2
def search_drug_knowledge(query: str, k: int = 4) -> dict:
    # Retrieve authoritative drug label passages for grounding an answer.
    hits = retrieve(query, k=int(k))
    return {'query': query, 'passages': [
        {'citation_id': f'S{i+1}', 'generic': h['generic'], 'source': h['source'],
         'section': h['title'], 'text': h['text'][:1200], 'relevance': round(h['score'], 3)}
        for i, h in enumerate(hits)]}


# ---------------------------------------------------------------- TOOL 3
def check_interactions(medicines: list) -> dict:
    # Pairwise interaction + duplicate-active-ingredient check. Pure lookup, no LLM.
    resolved, unknown = [], []
    for m in medicines:
        r = resolve_medicine(m)
        if r.get('generic'):
            resolved.append({'input': m, 'generic': r['generic'],
                             'ingredients': r.get('active_ingredients', [r['generic']])})
        else:
            unknown.append(m)

    gens = [r['generic'] for r in resolved]

    # Match on the full generic AND on each active ingredient, so a combination
    # product still triggers rules written against its components: Panadol Extra
    # is 'paracetamol + caffeine', but the warfarin rule is keyed on 'paracetamol'.
    for r in resolved:
        r['terms'] = list(dict.fromkeys(
            [r['generic']] + [i.strip() for i in r['ingredients']]))

    found = []
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            seen_pairs = set()
            for a in resolved[i]['terms']:
                for b in resolved[j]['terms']:
                    if a == b:
                        continue            # same ingredient - that is a duplicate, handled below
                    hit = inter_df[((inter_df.generic_a == a) & (inter_df.generic_b == b)) |
                                   ((inter_df.generic_a == b) & (inter_df.generic_b == a))]
                    for row in hit.itertuples():
                        key = tuple(sorted((a, b)))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        found.append({'drug_a': resolved[i]['input'],
                                      'drug_b': resolved[j]['input'],
                                      'generics': f'{a} + {b}', 'severity': row.severity,
                                      'mechanism': row.mechanism, 'advice': row.advice})

    # duplicate active ingredients - the paracetamol stacking problem
    seen, dupes = {}, []
    for r in resolved:
        for ing in r['ingredients']:
            seen.setdefault(ing.strip().lower(), []).append(r['input'])
    for ing, sources in seen.items():
        if len(sources) > 1:
            dupes.append({'ingredient': ing, 'appears_in': sources, 'severity': 'major',
                          'advice': f'{len(sources)} of these products contain {ing}. Taking them '
                                    f'together stacks the dose and can cause overdose without the '
                                    f'patient realising it.'})

    sev_rank = {'major': 0, 'moderate': 1, 'minor': 2}
    found.sort(key=lambda x: sev_rank.get(x['severity'], 9))
    return {'medicines_checked': gens, 'unrecognised': unknown,
            'interactions': found, 'duplicate_ingredients': dupes,
            'summary': f'{len(found)} interaction(s), {len(dupes)} duplicate-ingredient issue(s).'}


# ---------------------------------------------------------------- TOOL 4
def check_dose(medicine: str, dose_amount: float, doses_per_day: int) -> dict:
    # Compare a proposed regimen against curated maximum single and daily doses.
    r = resolve_medicine(medicine)
    gen = r.get('generic')
    if not gen:
        return {'medicine': medicine, 'verdict': 'unknown_medicine',
                'note': 'Could not identify the medicine, so no dose check was possible.'}

    # combination products: check the first named ingredient we hold a limit for
    key = gen if gen in MAX_DOSE else next(
        (i for i in r.get('active_ingredients', []) if i in MAX_DOSE), None)
    if key is None:
        return {'medicine': medicine, 'generic': gen, 'verdict': 'no_reference_data',
                'note': 'No maximum-dose reference is held locally. A pharmacist should confirm.'}

    ref = MAX_DOSE[key]
    if ref.max_daily_dose == 0:
        return {'medicine': medicine, 'generic': gen, 'verdict': 'clinician_directed',
                'note': ref.note}

    total = float(dose_amount) * int(doses_per_day)
    flags = []
    if float(dose_amount) > ref.max_single_dose:
        flags.append(f'Single dose {dose_amount} {ref.unit} exceeds the usual maximum of '
                     f'{ref.max_single_dose} {ref.unit}.')
    if total > ref.max_daily_dose:
        flags.append(f'Daily total {total:g} {ref.unit} exceeds the maximum of '
                     f'{ref.max_daily_dose} {ref.unit}.')

    return {'medicine': medicine, 'generic': gen, 'checked_against': key,
            'daily_total': f'{total:g} {ref.unit}', 'max_single': f'{ref.max_single_dose} {ref.unit}',
            'max_daily': f'{ref.max_daily_dose} {ref.unit}',
            'verdict': 'EXCEEDS_LIMIT' if flags else 'within_limits',
            'flags': flags, 'note': ref.note}


# ---------------------------------------------------------------- TOOL 5
def check_red_flags(symptom_text: str) -> dict:
    # Screen free text for symptoms that need a doctor now, not a medicine.
    t = str(symptom_text).lower()
    hits = []
    for row in flags_df.itertuples():
        for pat in row.pattern.split('|'):
            if pat.strip() in t:
                hits.append({'matched': pat.strip(), 'category': row.category,
                             'urgency': row.urgency, 'message_en': row.message_en,
                             'message_ur': row.message_ur})
                break
    order = {'emergency': 0, 'urgent': 1, 'caution': 2}
    hits.sort(key=lambda h: order.get(h['urgency'], 9))
    return {'red_flags_found': len(hits), 'flags': hits,
            'action': 'STOP AND ESCALATE - tell the user to seek medical care immediately, before '
                      'discussing any medicine.'
                      if any(h['urgency'] == 'emergency' for h in hits)
                      else 'proceed with normal guidance'}


TOOLS = {'resolve_medicine': resolve_medicine, 'search_drug_knowledge': search_drug_knowledge,
         'check_interactions': check_interactions, 'check_dose': check_dose,
         'check_red_flags': check_red_flags}
print('5 tools registered:', ', '.join(TOOLS))
# ============================================================================
# --- from notebook cell 20 ---
# ============================================================================
def _schema(props, required):
    return types.Schema(type=types.Type.OBJECT, properties=props, required=required)

S  = lambda d: types.Schema(type=types.Type.STRING, description=d)
N  = lambda d: types.Schema(type=types.Type.NUMBER, description=d)
I  = lambda d: types.Schema(type=types.Type.INTEGER, description=d)
AR = lambda d: types.Schema(type=types.Type.ARRAY, description=d,
                            items=types.Schema(type=types.Type.STRING))

DECLARATIONS = [
    types.FunctionDeclaration(
        name='check_red_flags',
        description='ALWAYS call this FIRST on any message describing symptoms. Screens for emergency '
                    'symptoms that need a doctor immediately rather than a medicine.',
        parameters=_schema({'symptom_text': S('The full text of what the user described.')},
                           ['symptom_text'])),
    types.FunctionDeclaration(
        name='resolve_medicine',
        description='Convert a Pakistani brand name (Panadol, Risek, Velosef, Augmentin...) into its '
                    'generic ingredient, drug class and prescription status. Tolerates typos. Call this '
                    'before reasoning about any named medicine.',
        parameters=_schema({'name': S('Brand or generic name as the user wrote it.')}, ['name'])),
    types.FunctionDeclaration(
        name='check_interactions',
        description='Check a list of medicines for drug-drug interactions AND duplicate active '
                    'ingredients (e.g. three cold remedies that all contain paracetamol). Call whenever '
                    'two or more medicines are mentioned.',
        parameters=_schema({'medicines': AR('All medicine names the patient takes, brand or generic.')},
                           ['medicines'])),
    types.FunctionDeclaration(
        name='check_dose',
        description='Check whether a proposed dose exceeds the maximum safe single or daily dose. Use '
                    'whenever the user states how much and how often they take something.',
        parameters=_schema({'medicine': S('Brand or generic name.'),
                            'dose_amount': N('Amount per dose in mg (mcg for levothyroxine).'),
                            'doses_per_day': I('How many times per day.')},
                           ['medicine', 'dose_amount', 'doses_per_day'])),
    types.FunctionDeclaration(
        name='search_drug_knowledge',
        description='Retrieve authoritative drug label passages to ground an explanation. Use for any '
                    'factual claim about what a medicine does, its side effects, cautions or dosing. '
                    'Returns passages with citation_id values you must cite.',
        parameters=_schema({'query': S('A focused question, e.g. "metronidazole alcohol interaction".'),
                            'k': I('How many passages to retrieve (default 4).')}, ['query'])),
]

SYSTEM_PROMPT = '''
You are MedSaathi, a medicine-safety assistant for patients in Pakistan. You are a careful pharmacist's
assistant, not a doctor.

## Non-negotiable rules
1. SAFETY FIRST. If the user describes symptoms, call `check_red_flags` before anything else. If any
   emergency flag fires, your entire reply is: what is worrying, and to seek medical care NOW. Do not
   discuss medicines in that reply.
2. NEVER DIAGNOSE. Do not name a condition the patient has. Describe what medicines do, not what is
   wrong with the person.
3. NEVER PRESCRIBE. Do not tell anyone to start, stop, or change a prescribed medicine. You may say
   "this combination is risky - ask your doctor or pharmacist before your next dose".
4. NEVER INVENT NUMBERS. Every dose figure must come from `check_dose` or a retrieved passage. If you do
   not have it, say you do not have it.
5. ALWAYS GROUND AND CITE. Any factual claim about a medicine must come from `search_drug_knowledge`.
   Cite inline as [S1], [S2] matching the citation_id of the passage you used.
6. ANTIBIOTICS. If the user mentions taking an antibiotic without a prescription, or stopping one early,
   note briefly why that drives antibiotic resistance.
7. PREGNANCY, BREASTFEEDING, INFANTS, KIDNEY OR LIVER DISEASE: flag that a clinician must review every
   medicine, and do not give dosing guidance for children.

## How to work
- Resolve every brand name with `resolve_medicine` first - the user will write "Risek", not "omeprazole".
- Whenever two or more medicines are named, call `check_interactions`. Always.
- Chain tools freely: resolve -> interactions -> retrieve -> explain. Do not answer from memory.
- If a tool returns nothing useful, say so plainly rather than filling the gap with guesses.

## Output format
**⚠️ Immediate concerns** - only if something is major or an emergency. Otherwise omit this section.
**Your medicines** - one short line each: brand → generic → what it is for [S#].
**What to watch** - interactions, duplicates, dose problems, in plain language, with the reason why.
**What to do** - a concrete next step, e.g. "take these two 2 hours apart", "ask your pharmacist before
the next dose", "this needs a doctor today".
**Sources** - the citation ids you used and their section names.

Close every answer with: *MedSaathi is an educational tool. It does not replace a doctor or pharmacist.*

Write at a 10th-grade reading level. Short sentences. If the user asks for Urdu, reply fully in Urdu
(Roman Urdu if they wrote in Roman Urdu), keeping the same structure and the same citations.
'''

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=DECLARATIONS)],
    temperature=0.2,
)
print('agent configured with', len(DECLARATIONS), 'tools')
# ============================================================================
# --- from notebook cell 21 ---
# ============================================================================
# >>> ACTION REQUIRED BEFORE YOU DEMO <<<
# Put a crisis helpline you have PERSONALLY VERIFIED is currently correct here,
# e.g. 'Umang Pakistan 0311-XXXXXXX'. Leave it as None rather than guessing:
# an out-of-date crisis number is worse than no number at all.
CRISIS_HELPLINE = '1122'


def run_agent(user_text, image_bytes=None, language='English', history=None,
              max_steps=6, verbose=True):
    # Manual tool-calling loop. Returns (answer, trace, updated_history).
    CRISIS_REPLY = (
        '**⚠️ Please reach out for support right now.**\n\n'
        'I cannot help with this, but you should not be dealing with it alone. Please talk to '
        'someone you trust and go to your nearest hospital emergency department.\n\n'
        + (f'You can also contact: **{CRISIS_HELPLINE}**\n\n' if CRISIS_HELPLINE else '') +
        'If someone has already taken too many tablets, take them to a hospital immediately, '
        'even if they seem fine.\n\n'
        '*MedSaathi is an educational tool. It does not replace a doctor or pharmacist.*')

    contents = list(history or [])
    parts = []
    if image_bytes:
        try:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'))
        except Exception:
            parts.append(types.Part(inline_data=types.Blob(data=image_bytes, mime_type='image/jpeg')))
        parts.append(types.Part(text='This is a photo of medicine packaging or a prescription. '
                                     'Read every medicine name and strength you can see.'))
    parts.append(types.Part(text=f'[Respond in {language}]\n\n{user_text}'))
    contents.append(types.Content(role='user', parts=parts))

    trace = []
    for step in range(max_steps):
        resp = _gen(model=CHAT_MODEL, contents=contents, config=CONFIG)

        # Gemini's safety filters can return an empty candidate on self-harm or
        # sensitive medical prompts. Never let that crash the loop mid-demo.
        cand = resp.candidates[0] if getattr(resp, 'candidates', None) else None
        if cand is None or cand.content is None or not (cand.content.parts or []):
            reason = str(getattr(cand, 'finish_reason', '') or
                         getattr(getattr(resp, 'prompt_feedback', None), 'block_reason', '') or
                         'unknown')
            if verbose:
                print(f'  [step {step+1}] model returned no content (finish_reason={reason})')
            if any(k in reason.upper() for k in ('SAFETY', 'BLOCK', 'PROHIBIT')):
                return CRISIS_REPLY, trace, contents
            return ('I could not produce a safe answer to that. Please ask a pharmacist or doctor '
                    'to review these medicines.'), trace, contents

        contents.append(cand.content)
        calls = [p.function_call for p in (cand.content.parts or [])
                 if getattr(p, 'function_call', None)]
        if not calls:
            answer = ''.join(p.text for p in (cand.content.parts or []) if getattr(p, 'text', None))
            return answer, trace, contents

        reply_parts = []
        for fc in calls:
            args = dict(fc.args or {})
            if verbose:
                print(f'  [step {step+1}] -> {fc.name}({json.dumps(args, default=str)[:110]})')
            try:
                result = TOOLS[fc.name](**args)
            except Exception as e:
                result = {'error': f'{type(e).__name__}: {e}'}
            trace.append({'step': step + 1, 'tool': fc.name, 'args': args, 'result': result})
            reply_parts.append(types.Part.from_function_response(name=fc.name,
                                                                 response={'result': result}))
        contents.append(types.Content(role='user', parts=reply_parts))

    return ('I could not finish the safety check within the step limit. Please ask a pharmacist '
            'to review these medicines.'), trace, contents


print('agent loop ready')
# ============================================================================
# --- from notebook cell 36 (Gradio UI), adapted for Render ---
# ============================================================================
import gradio as gr

# Gradio's API moves between major versions: v6 removed Chatbot(type=...) and
# moved theme/css from Blocks() to launch(). Rather than pin a version, drop any
# keyword this install does not accept and carry on.
GRADIO_MAJOR = int(gr.__version__.split('.')[0])
print('gradio version', gr.__version__, flush=True)

def build(fn, **kw):
    while True:
        try:
            return fn(**kw)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([A-Za-z_]+)'", str(e))
            if not m or m.group(1) not in kw:
                raise
            kw.pop(m.group(1))
            print(f'   (gradio {gr.__version__} ignores {m.group(1)!r} - dropped)', flush=True)


EXAMPLES = [
    ['My father takes Warfarin for his heart. He is also taking Panadol Extra, Flu-Out and '
     'Brufen 400mg three times a day for fever. Is this safe?'],
    ['Can I take Panadol, Panadol Extra and Flu-Out together for flu?'],
    ['Mujhe pait mein dard hai aur main Risek aur Ponstan sath le raha hun. Kya ye theek hai?'],
    ['I take Panadol 1000mg five times a day for back pain. Is that safe?'],
    ['Doctor gave me Augmentin. Can I stop after 3 days if I feel better?'],
]


def format_summary(trace):
    # A colour-coded banner of what the safety tools actually found.
    majors, moderates, dupes, emergencies, doses, meds = [], [], [], [], [], []
    for t in trace:
        res, tool = t['result'], t['tool']
        if tool == 'check_interactions':
            for i in res.get('interactions', []):
                (majors if i['severity'] == 'major' else moderates).append(i['generics'])
            dupes += [d['ingredient'] for d in res.get('duplicate_ingredients', [])]
            meds += res.get('medicines_checked', [])
        elif tool == 'check_red_flags':
            emergencies += [f['category'] for f in res.get('flags', [])
                            if f['urgency'] == 'emergency']
        elif tool == 'check_dose' and res.get('verdict') == 'EXCEEDS_LIMIT':
            doses.append(res.get('generic', '?'))
        elif tool == 'resolve_medicine' and res.get('generic'):
            meds.append(res['generic'])

    if not trace:
        return "<div class='ms-card ms-idle'>Ask a question to run a safety check.</div>"

    chips = []
    if emergencies:
        chips.append(('crit', 'EMERGENCY - seek care now (' +
                      ', '.join(sorted(set(emergencies))[:3]) + ')'))
    for d in sorted(set(dupes)):
        chips.append(('crit', f'Duplicate ingredient: {d}'))
    for m in sorted(set(majors)):
        chips.append(('crit', f'Major interaction: {m}'))
    for m in sorted(set(moderates)):
        chips.append(('warn', f'Moderate interaction: {m}'))
    for d in sorted(set(doses)):
        chips.append(('warn', f'Dose above safe maximum: {d}'))
    if not chips:
        chips.append(('ok', 'No interactions or dose problems found in our database'))

    med_line = ''
    if meds:
        pills = ''.join(f"<span class='ms-pill'>{m}</span>" for m in sorted(set(meds))[:10])
        med_line = f"<div class='ms-meds'><b>Identified:</b> {pills}</div>"

    body = ''.join(f"<div class='ms-chip ms-{k}'>{v}</div>" for k, v in chips)
    return f"<div class='ms-card'>{body}{med_line}</div>"


def respond(message, image, language, chat_history):
    if not (message or '').strip() and image is None:
        return chat_history, '', format_summary([])
    img_bytes = None
    if image is not None:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(image).convert('RGB').save(buf, format='JPEG', quality=85)
        img_bytes = buf.getvalue()
    try:
        answer, trace, _ = run_agent(message or 'What are these medicines?',
                                     image_bytes=img_bytes, language=language, verbose=False)
    except Exception as e:
        # Say what actually happened. A demo that hides its error is impossible to
        # fix while a room is watching.
        msg, trace = str(e), []
        if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
            answer = ('**Out of free-tier Gemini quota.**\n\n'
                      'Every model on this API key has hit its free limit. The per-minute '
                      'limit clears in about a minute; the daily one resets at midnight US '
                      'Pacific.\n\n'
                      'Nothing is broken: retrieval and all five safety checks run locally '
                      'and still work. Only the model that writes the explanation is capped.')
        else:
            answer = (f'**Something went wrong:** `{type(e).__name__}`\n\n'
                      f'```\n{msg[:400]}\n```')
    chat_history = (chat_history or []) + [
        {'role': 'user', 'content': message or '(photo)'},
        {'role': 'assistant', 'content': answer}]
    return chat_history, '', format_summary(trace)


CSS = '''
/* Fit the whole app on one screen - a demo that scrolls loses the room. */
.gradio-container {max-width:100% !important; padding:10px 18px 0 !important;}
footer {display:none !important;}
.ms-side {max-height:72vh; overflow-y:auto; padding-right:4px;}
/* The chat box is sized against the viewport, not a fixed pixel count, so the
   input row and the disclaimer stay on screen on a small laptop too. */
#ms-chat, #ms-chat .wrap, #ms-chat .wrapper, #ms-chat .bubble-wrap {
  height: calc(100vh - 490px) !important; min-height:170px !important;
  max-height:480px !important;}

.ms-hero {background: linear-gradient(135deg,#2563eb 0%,#1d4ed8 55%,#1e40af 100%);
          color:#fff; padding:14px 22px; border-radius:14px; margin-bottom:8px;}
.ms-hero h1 {margin:0 0 2px 0; font-size:25px; font-weight:700; color:#fff;}
.ms-hero p {margin:0; opacity:.92; font-size:14px; color:#fff;}
/* The banner wraps into columns and scrolls inside itself, so a case with five
   findings cannot push the input box off the bottom of the screen. */
.ms-card {display:flex; flex-wrap:wrap; gap:6px; max-height:132px; overflow-y:auto;
          align-content:flex-start; padding-bottom:2px;}
.ms-chip {flex:1 1 270px; padding:8px 12px; border-radius:9px; font-weight:600;
          font-size:13.5px; border-left:4px solid;}
.ms-crit {background:#fef2f2; color:#991b1b; border-color:#dc2626;}
.ms-warn {background:#fffbeb; color:#92400e; border-color:#f59e0b;}
.ms-ok   {background:#eff6ff; color:#1e40af; border-color:#2563eb;}
.ms-idle {background:#f8fafc; color:#64748b; padding:10px 14px; border-radius:10px;}
.ms-meds {flex-basis:100%; margin-top:2px; font-size:13px; color:#334155;}
.ms-pill {display:inline-block; background:#dbeafe; color:#1d4ed8; border-radius:999px;
          padding:2px 10px; margin:2px 4px 0 0; font-size:12px; font-weight:600;}
.ms-foot {text-align:center; color:#64748b; font-size:12px; margin:8px 0 4px;}
@media (prefers-color-scheme: dark) {
  .ms-crit{background:#450a0a; color:#fecaca;} .ms-warn{background:#451a03; color:#fde68a;}
  .ms-ok{background:#172554; color:#bfdbfe;} .ms-idle{background:#1e293b; color:#94a3b8;}
  .ms-meds{color:#cbd5e1;} .ms-pill{background:#1e3a8a; color:#bfdbfe;}
}
'''

# v6 wants theme/css at launch(); earlier versions want them on Blocks().
_style = dict(theme=gr.themes.Soft(primary_hue='blue'), css=CSS)
_blocks_kw = dict(title='MedSaathi')
_launch_kw = dict()
(_launch_kw if GRADIO_MAJOR >= 6 else _blocks_kw).update(_style)

demo = build(gr.Blocks, **_blocks_kw)
with demo:
    gr.HTML("<div class='ms-hero'><h1>MedSaathi</h1>"
            "<p>Medicine safety checks for Pakistan - brand names, interactions, "
            "and plain-language answers in English or Urdu</p></div>")

    with gr.Row():
        # ---- left: the conversation ---------------------------------------
        with gr.Column(scale=3):
            summary = gr.HTML(format_summary([]))
            # 'type' only exists before v6; v6 uses the messages format by default,
            # which is what our {'role','content'} dicts already are.
            _chat_kw = dict(height=312, label='MedSaathi', show_copy_button=True,
                     elem_id='ms-chat')
            if GRADIO_MAJOR < 6:
                _chat_kw['type'] = 'messages'
            chat = build(gr.Chatbot, **_chat_kw)
            with gr.Row():
                box = build(gr.Textbox,
                            placeholder='e.g. I take Risek and Ponstan together - is that safe?',
                            scale=5, show_label=False, lines=2, autofocus=True)
                send = gr.Button('Check safety', variant='primary', scale=1)

        # ---- right: the controls, beside the chat instead of underneath ----
        with gr.Column(scale=2, elem_classes='ms-side'):
            with gr.Row():
                lang = gr.Radio(['English', 'Urdu'], value='English',
                                label='Answer language', scale=3)
                clear = gr.Button('Clear', scale=1)
            with gr.Accordion('Or photograph the medicine strips', open=False):
                img = build(gr.Image, label='Printed strips work better than handwriting',
                            height=150)
            build(gr.Examples, examples=EXAMPLES, inputs=[box],
                  label='Try one of these', examples_per_page=5)

    gr.HTML("<div class='ms-foot'>Educational project. MedSaathi does not diagnose or "
            "prescribe, and does not replace a doctor or pharmacist. "
            "In an emergency, go to a hospital.</div>")

    _io = ([box, img, lang, chat], [chat, box, summary])
    send.click(respond, *_io)
    box.submit(respond, *_io)
    clear.click(lambda: ([], '', format_summary([])), None, [chat, box, summary])

PORT = int(os.environ.get('PORT', 7860))
print(f'launching on 0.0.0.0:{PORT}', flush=True)
demo.queue().launch(server_name='0.0.0.0', server_port=PORT)