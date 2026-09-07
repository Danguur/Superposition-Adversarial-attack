# Sample Attackability Framework for Spoken Language Processing

A framework for extending **sample attackability** — the idea that some inputs are inherently
easier to adversarially attack than others — from the image/text domains to speech, using
adversarial perturbations against [OpenAI Whisper](https://github.com/openai/whisper) that aim
to suppress transcription of an utterance ("muting").

This repository contains the experimental code, evaluation pipeline, and results for a pilot
study evaluating a **gradient-based superposition attack** on Whisper, with controlled
comparisons against random-noise baselines to rule out confounds such as perturbation
duration and amplitude.

> **Status:** Pilot / work in progress. Results below are from a small (39-utterance) pilot
> corpus against a single Whisper variant (`tiny.en`). See [Limitations](#limitations-and-honest-caveats)
> before drawing broader conclusions.

---

## Contents

```
.
├── superposition_attack_v2.py   # Main attack: gradient-based (MI-FGSM style) universal
│                                 # perturbation, with random-noise controls, duration-matched
│                                 # conditions, and full evaluation (confusion matrix, P/R/F1, AUROC)
├── muting_attack_v2.py          # Prepended-noise muting attack (corrected loss objective),
│                                 # with diagnostic checks and full evaluation
├── audio_files/                 # Input corpus (39 gTTS-synthesised short English phrases) -- not
│                                 # included in this repo; see Data below
├── audiowithnoise/               # Output: attacked audio saved per condition, for listening/demo
│   ├── clean/ , clean_demo/
│   ├── random_noise_30s/ , random_noise_30s_demo/
│   ├── trained_noise_30s/ , trained_noise_30s_demo/
│   ├── random_noise_5s/ , random_noise_5s_demo/
│   └── trained_noise_5s/ , trained_noise_5s_demo/
├── superposition_results.csv     # Per-sample results (condition, file, muted, no_speech_prob)
├── muting_results.csv            # Per-sample results for the muting attack
├── attack_segment.pt             # Saved learned muting-attack noise segment
├── superposition_noise_advanced.pt # Saved learned superposition perturbation
└── README.md
```

---

## Background

Modern spoken language processing (SLP) systems, including large sequence-to-sequence
transcribers such as Whisper, are known to be vulnerable to adversarial audio perturbations.
Most existing work treats every input utterance as equally susceptible to attack. This project
instead asks: **can we predict which utterances are inherently more "attackable" than others,
without running an expensive adversarial search on every sample?**

To answer this, the project:

1. Defines an **attackability score** for a speech sample as the smallest perturbation
   amplitude bound needed to make Whisper produce an empty transcript (a "muting" attack).
2. Evaluates two attack strategies that instantiate this objective:
   - **Muting attack** — a short (0.64 s) perturbation prepended to the clean audio.
   - **Superposition attack** — a universal perturbation added elementwise across the
     full input buffer, optimised via momentum iterative sign gradient ascent (MI-FGSM style)
     on Whisper's own `no_speech` token probability.
3. Designs (but has not yet trained) a lightweight **attackability detector** that would
   predict attackability directly from frozen Whisper encoder representations, without running
   an attack at inference time.

---

## Requirements

```bash
pip install torch numpy soundfile scikit-learn
pip install git+https://github.com/openai/whisper.git
```

Tested with Python 3.10+, `openai-whisper`, and both CPU and CUDA execution (CPU is
significantly slower — see [Limitations](#limitations-and-honest-caveats)).

---

## Usage

### Superposition attack (main result)

```bash
python superposition_attack_v2.py
```

This will:
1. Load Whisper `tiny.en` and all `.wav` files in `audio_files/`.
2. Train a universal perturbation via MI-FGSM-style optimisation (400 steps).
3. Evaluate the trained perturbation and a matched random-noise control under two conditions:
   perturbation spanning the full 30-second input buffer, and perturbation confined to a
   duration-matched first-5-second window.
4. Save attacked audio (full-length and short 6-second demo clips) to `audiowithnoise/`.
5. Print and save (`superposition_results.csv`) a full confusion matrix, precision, recall,
   F1 score, and AUROC (using Whisper's real `no_speech_prob`) for each condition.

### Muting attack (corrected, ongoing evaluation)

```bash
python muting_attack_v2.py
```

Trains and evaluates the prepended-noise muting attack, including a diagnostic that checks
whether Whisper's own `SuppressBlank` / timestamp-forcing decode rules are blocking the attack
independent of perturbation quality (see [Known Issues](#known-issues--lessons-learned)).

---

## Data

The pilot corpus consists of **39 short English phrases** (1–3 seconds each), synthesised with
Google Text-to-Speech (gTTS), covering common conversational expressions ("hello", "good
morning", "can you help me", etc.). The intended primary corpus was
[LibriSpeech](https://www.openslr.org/12), but this could not be downloaded in the pilot
environment; this substitution is a known limitation (see below).

---

## Results

**Attack success rate** (n = 39, Whisper `tiny.en`):

| Condition | Perturbation extent | Success rate |
|---|---|---|
| Random noise | Full 30 s | 0.0% |
| **Trained attack** | Full 30 s | **100.0%** |
| Random noise | Confined to first 5 s | 0.0% |
| **Trained attack** | Confined to first 5 s | **66.7%** |

The random-noise control achieved 0% success in both conditions, ruling out perturbation
duration or raw amplitude as the explanation for the trained attack's success — the effect is
attributable to the learned perturbation itself. The 5-second (duration-matched) condition is
the more realistic, conservative result; the full-30s condition is an idealised upper bound that
benefits from extra silent padding not present in a real deployment.

**Precision / recall / F1 / AUROC** (paired clean-vs-attacked classification):

| Condition | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Random noise, full 30s | 0.00 | 0.00 | 0.00 | 0.953 |
| Trained attack, full 30s | 1.00 | 1.00 | 1.00 | 1.000 |
| Random noise, confined 5s | 0.00 | 0.00 | 0.00 | 0.821 |
| Trained attack, confined 5s | 1.00 | 0.67 | 0.80 | 1.000 |

Notably, the random-noise control shows AUROC well above chance (0.82–0.95) despite 0%
behavioural success — even untrained noise measurably shifts Whisper's internal no-speech
confidence, but only the trained perturbation reliably crosses the decision threshold.

---

## Known Issues / Lessons Learned

These are documented honestly because they were genuinely useful findings during development:

- **Train/evaluation objective mismatch (muting attack).** An earlier version of the muting
  attack trained against a raw end-of-text logit assuming a bare start token, while
  `model.transcribe()` never evaluates that logit at that position — it's masked by Whisper's
  own `SuppressBlank` and timestamp-forcing decode rules. This produced a misleading result
  (training loss → 0, black-box success stuck at 0%). Fixed by teacher-forcing the real decode
  sequence and targeting the `no_speech` token instead.
- **Padding confound (superposition attack).** Whisper's encoder requires a fixed 30-second
  input; naively adding a perturbation across the whole buffer risks the effect being driven by
  noise duration rather than the learned pattern. Addressed with a duration-matched control
  (perturbation confined to 5 seconds, rest left as natural silence) and a random-noise
  baseline evaluated identically.
- **English-only model + language auto-detection.** `whisper.decode()` attempts language
  ID by default, which fails outright on `.en` (English-only) checkpoints since they have no
  language tokens. Fix: always pass `language="en"` explicitly in `DecodingOptions` when using
  an `.en` model.

## Limitations and Honest Caveats

- Pilot corpus is small (39 utterances) and synthetic (gTTS), not naturally recorded speech.
- Only Whisper `tiny.en` has been evaluated; `base.en` / `small.en` / `medium.en` remain planned.
- The attackability detector architecture is designed and implemented but **has not yet been
  trained or evaluated** — this repository reports attack results only, not detector
  performance.
- All experiments were run on CPU, which limited the optimisation budget (steps/epochs) used.

---

## Related Work

This project is directly motivated by prior work on universal acoustic adversarial attacks
against Whisper, in particular the prepended end-of-text-token attack described in:

> V. Raina, R. Ma, C. McGhee, K. Knill, M. Gales. *Muting Whisper: A Universal Acoustic
> Adversarial Attack on Speech Foundation Models*. arXiv:2405.06134, 2024.

See the accompanying paper draft in this project for a full literature review and additional
citations.

---

## Author

Zahoor Ahmad Dar

## Contact us 

If there is any error you can contact us at zahoor.phd.it.2026@cusrinagar.edu.in
