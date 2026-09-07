import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import whisper
import soundfile as sf
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
AUDIO_FOLDER = "audio_files"
NOISY_AUDIO_FOLDER = "audiowithnoise"   # where attacked .wav files get saved for demo/listening
DEMO_TRIM_SECONDS = 6                   # short, easy-to-listen-to preview length
DEMO_TRIM_SAMPLES = DEMO_TRIM_SECONDS * 16000
MODEL_NAME = "tiny.en"
AMPLITUDE_LIMIT = 0.05          # same L-inf budget as the original script
NUM_SAMPLES = 480_000           # 30s @ 16kHz  (ORIGINAL padding — kept for comparability)
NUM_SAMPLES_SHORT = 80_000      # 5s @ 16kHz   (duration-matched control padding)
BATCH_PER_STEP = 6              # files used per gradient step (universal perturbation)
STEPS = 400                     # gradient steps
STEP_SIZE = 0.002               # PGD step size (in waveform amplitude units)
MOMENTUM_DECAY = 0.9
EOT_SAMPLES = 3                 # augmentations averaged per step for robustness
EVAL_EVERY = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_CSV = "superposition_results.csv"

print("=" * 60)
print("SUPERPOSITION ATTACK (ADVANCED - gradient-based)")
print("=" * 60)
print(f"Device: {DEVICE}")

model = whisper.load_model(MODEL_NAME).to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

tokenizer = whisper.tokenizer.get_tokenizer(
    model.is_multilingual, num_languages=model.num_languages, task="transcribe"
)

audio_files = [
    os.path.join(AUDIO_FOLDER, f)
    for f in os.listdir(AUDIO_FOLDER)
    if f.endswith(".wav")
]
print(f"Found {len(audio_files)} audio files")
assert len(audio_files) > 0, "No .wav files found in audio_files/"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_audio_tensor(path: str, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """Load + pad/trim to num_samples, return float32 tensor on DEVICE."""
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio, num_samples)
    return torch.from_numpy(audio.astype(np.float32)).to(DEVICE)


def eot_augment(waveform: torch.Tensor) -> torch.Tensor:
    """Random gain + circular shift, differentiable, for EOT robustness."""
    gain = 1.0 + (torch.rand(1, device=DEVICE).item() - 0.5) * 0.2
    shift = random.randint(-1600, 1600)
    out = waveform * gain
    if shift != 0:
        out = torch.roll(out, shifts=shift, dims=-1)
    return out


def no_speech_loss(waveform: torch.Tensor) -> torch.Tensor:
    """Differentiable proxy loss: negative log-prob of <|nospeech|>."""
    mel = whisper.log_mel_spectrogram(waveform, n_mels=model.dims.n_mels).unsqueeze(0)
    mel = mel.to(DEVICE)
    audio_features = model.encoder(mel)
    sot_sequence = torch.tensor(
        [list(tokenizer.sot_sequence_including_notimestamps)], device=DEVICE
    )
    logits = model.decoder(sot_sequence, audio_features)
    first_step_logits = logits[0, -1]
    log_probs = F.log_softmax(first_step_logits, dim=-1)
    return -log_probs[tokenizer.no_speech]


def transcribes_empty(waveform_np: np.ndarray) -> bool:
    """Black-box check: does model.transcribe() return an empty string?"""
    result = model.transcribe(waveform_np.astype(np.float32))
    return result["text"].strip() == ""


def get_no_speech_prob(waveform_np: np.ndarray) -> float:
    """
    Real no_speech_prob from Whisper's own decoder (via whisper.decode),
    used as a continuous score for AUROC. This is the model's own
    calibrated probability that the clip contains no speech, distinct
    from the raw training-time logit trick used in no_speech_loss.
    """
    mel = whisper.log_mel_spectrogram(waveform_np.astype(np.float32), n_mels=model.dims.n_mels)
    mel = mel.to(DEVICE)
    options = whisper.DecodingOptions(
        fp16=(DEVICE == "cuda"), without_timestamps=True, language="en"
    )
    result = whisper.decode(model, mel, options)
    # whisper.decode returns a single DecodingResult (or list); handle both
    if isinstance(result, list):
        result = result[0]
    return float(result.no_speech_prob)


def build_condition(file_paths, noise_full=None, condition_name="condition", save_audio=True,
                     sample_rate=16000):
    """
    Always pads/trims audio to the full NUM_SAMPLES (30s), since Whisper's
    encoder has a fixed-size positional embedding and CANNOT accept a
    shorter input -- that's what caused the 'incorrect audio shape' error.
    `noise_full` must already be a length-NUM_SAMPLES array (use
    zero-padding, not truncation, to create a "duration-matched" noise
    condition -- see make_duration_limited_noise below).

    If save_audio=True, every attacked waveform is written to
    NOISY_AUDIO_FOLDER/<condition_name>/<original_filename>.wav so the
    actual noisy audio can be listened to / used for demonstration.
    """
    rows = []
    if save_audio:
        out_dir = os.path.join(NOISY_AUDIO_FOLDER, condition_name)
        os.makedirs(out_dir, exist_ok=True)

    for f in file_paths:
        audio = whisper.load_audio(f)
        audio = whisper.pad_or_trim(audio, NUM_SAMPLES)
        if noise_full is not None:
            attacked = np.clip(audio + noise_full, -1, 1).astype(np.float32)
        else:
            attacked = audio.astype(np.float32)

        empty = transcribes_empty(attacked)
        nsp = get_no_speech_prob(attacked)
        rows.append({"file": os.path.basename(f), "muted": int(empty), "no_speech_prob": nsp})

        if save_audio:
            out_path = os.path.join(out_dir, os.path.basename(f))
            sf.write(out_path, attacked, sample_rate)

            # Also save a short, easy-to-listen-to preview (real speech +
            # noise only, no long silent tail) so playback isn't 30s of
            # near-silence -- this is purely for demonstration/listening.
            demo_dir = os.path.join(NOISY_AUDIO_FOLDER, condition_name + "_demo")
            os.makedirs(demo_dir, exist_ok=True)
            demo_clip = attacked[:DEMO_TRIM_SAMPLES]
            sf.write(os.path.join(demo_dir, os.path.basename(f)), demo_clip, sample_rate)

    if save_audio:
        print(f"  💾 Saved {len(file_paths)} attacked audio files to {out_dir}/")
        print(f"  💾 Saved {len(file_paths)} short demo clips ({DEMO_TRIM_SECONDS}s) to {demo_dir}/")

    return rows


def make_duration_limited_noise(noise_short, total_len=NUM_SAMPLES):
    """
    Places a short noise segment (e.g. 5s worth of samples) at the start of
    a full-length (30s) buffer and zeros out the rest. This tests whether
    the ATTACK's effect depends on noise being spread across a long silent
    padding region, WITHOUT changing the model's required input length.
    """
    full = np.zeros(total_len, dtype=np.float32)
    full[:len(noise_short)] = noise_short
    return full


def make_random_noise(num_samples, amplitude, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(-amplitude, amplitude, num_samples).astype(np.float32)


# ----------------------------------------------------------------------
# Initialize + train universal perturbation (original 30s / 480,000-sample
# padding is kept exactly as before, for comparability with earlier runs)
# ----------------------------------------------------------------------
noise = (torch.randn(NUM_SAMPLES, device=DEVICE) * 0.01).clamp(
    -AMPLITUDE_LIMIT, AMPLITUDE_LIMIT
)
noise.requires_grad_(True)
momentum = torch.zeros_like(noise)

print("Preloading audio into memory...")
waveforms = [load_audio_tensor(p, NUM_SAMPLES) for p in audio_files]

print("\nTraining (gradient-based PGD, universal perturbation)...")

for step in range(1, STEPS + 1):
    batch_idx = random.sample(range(len(waveforms)), k=min(BATCH_PER_STEP, len(waveforms)))
    total_loss = 0.0
    grad_accum = torch.zeros_like(noise)

    for idx in batch_idx:
        clean = waveforms[idx]
        for _ in range(EOT_SAMPLES):
            perturbed = torch.clamp(eot_augment(clean) + noise, -1.0, 1.0)
            loss = no_speech_loss(perturbed)
            grad = torch.autograd.grad(loss, noise, retain_graph=False)[0]
            grad_accum += grad
            total_loss += loss.item()

    grad_accum /= (len(batch_idx) * EOT_SAMPLES)
    grad_norm = grad_accum / (grad_accum.abs().mean() + 1e-12)
    momentum = MOMENTUM_DECAY * momentum + grad_norm

    with torch.no_grad():
        noise -= STEP_SIZE * momentum.sign()
        noise.clamp_(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)

    if step % EVAL_EVERY == 0 or step == 1:
        avg_loss = total_loss / (len(batch_idx) * EOT_SAMPLES)
        with torch.no_grad():
            noise_np = noise.detach().cpu().numpy()
            sample_files = audio_files[:10]
            success = 0
            for f in sample_files:
                audio = whisper.load_audio(f)
                audio = whisper.pad_or_trim(audio, NUM_SAMPLES)
                attacked = np.clip(audio + noise_np[:len(audio)], -1, 1).astype(np.float32)
                if transcribes_empty(attacked):
                    success += 1
            rate = success / len(sample_files) * 100
        print(f"Step {step:4d} | no_speech_loss={avg_loss:.4f} | "
              f"sample success={rate:.1f}% ({success}/{len(sample_files)})")

noise_np_trained = noise.detach().cpu().numpy()
torch.save({"noise": torch.from_numpy(noise_np_trained), "amplitude_limit": AMPLITUDE_LIMIT},
           "superposition_noise_advanced.pt")
print("✅ Trained noise saved to superposition_noise_advanced.pt")

# ----------------------------------------------------------------------
# FULL EVALUATION SECTION
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("FULL EVALUATION: confusion matrix, precision/recall/F1, AUROC")
print("=" * 60)

# Fixed random-noise control, same amplitude, generated once (reproducible)
random_noise_30s_full = make_random_noise(NUM_SAMPLES, AMPLITUDE_LIMIT, seed=0)
random_noise_5s_short = random_noise_30s_full[:NUM_SAMPLES_SHORT]
random_noise_5s_full = make_duration_limited_noise(random_noise_5s_short)

trained_noise_5s_short = noise_np_trained[:NUM_SAMPLES_SHORT]
trained_noise_5s_full = make_duration_limited_noise(trained_noise_5s_short)

conditions = {
    "clean":                dict(noise=None),
    "random_noise_30s":     dict(noise=random_noise_30s_full),
    "trained_noise_30s":    dict(noise=noise_np_trained),
    "random_noise_5s":      dict(noise=random_noise_5s_full),   # noise only in first 5s, rest silent
    "trained_noise_5s":     dict(noise=trained_noise_5s_full),  # noise only in first 5s, rest silent
}

results = {}
os.makedirs(NOISY_AUDIO_FOLDER, exist_ok=True)
for name, cfg in conditions.items():
    print(f"\nRunning condition: {name} ...")
    rows = build_condition(audio_files, cfg["noise"], condition_name=name, save_audio=True)
    results[name] = rows
    rate = 100.0 * sum(r["muted"] for r in rows) / len(rows)
    print(f"  -> muted rate: {rate:.1f}% ({sum(r['muted'] for r in rows)}/{len(rows)})")

# ---- Write per-sample CSV for transparency ----
with open(RESULTS_CSV, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["condition", "file", "muted", "no_speech_prob"])
    for name, rows in results.items():
        for r in rows:
            writer.writerow([name, r["file"], r["muted"], f"{r['no_speech_prob']:.6f}"])
print(f"\n✅ Per-sample results written to {RESULTS_CSV}")


def paired_metrics(clean_rows, attacked_rows, label):
    """
    Build a labelled dataset: clean condition = label 0 (expected NOT muted),
    attacked condition = label 1 (expected muted). Prediction = actual muted
    outcome. Reports confusion matrix, precision/recall/F1, and AUROC using
    no_speech_prob as the continuous score.
    """
    y_true = [0] * len(clean_rows) + [1] * len(attacked_rows)
    y_pred = [r["muted"] for r in clean_rows] + [r["muted"] for r in attacked_rows]
    y_score = [r["no_speech_prob"] for r in clean_rows] + [r["no_speech_prob"] for r in attacked_rows]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")  # occurs if y_true has only one class present

    print(f"\n--- {label} ---")
    print("Confusion matrix (rows=true [0=clean,1=attacked], cols=pred [0=not muted,1=muted]):")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["not_muted_true", "muted_true"],
                                 zero_division=0))
    print(f"AUROC (no_speech_prob, clean vs attacked): {auroc:.4f}")

    return {
        "label": label,
        "confusion_matrix": cm.tolist(),
        "precision_class1": precision[1],
        "recall_class1": recall[1],
        "f1_class1": f1[1],
        "auroc": auroc,
    }


summary = []
summary.append(paired_metrics(results["clean"], results["random_noise_30s"],
                               "Random noise vs clean (noise spread across full 30s)"))
summary.append(paired_metrics(results["clean"], results["trained_noise_30s"],
                               "Trained attack vs clean (noise spread across full 30s)"))
summary.append(paired_metrics(results["clean"], results["random_noise_5s"],
                               "Random noise vs clean (noise confined to first 5s, duration-matched)"))
summary.append(paired_metrics(results["clean"], results["trained_noise_5s"],
                               "Trained attack vs clean (noise confined to first 5s, duration-matched)"))

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Condition':55s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUROC':>7s}")
for s in summary:
    print(f"{s['label']:55s} {s['precision_class1']:.3f}  {s['recall_class1']:.3f}  "
          f"{s['f1_class1']:.3f}  {s['auroc']:.3f}")

print("""
INTERPRETATION GUIDE:
- If 'random noise vs clean (full 30s)' already shows high recall/F1 and high
  AUROC, the 100% success rate seen with noise spread across all 30s is NOT
  specific to the trained perturbation -- it is likely an artefact of a short
  clip being drowned in a long region of added noise (the padding confound).
- The '(confined to first 5s)' conditions are the fairer test: noise only
  overlaps roughly where the real speech is, with the remaining ~25s left as
  natural silence (matching how pad_or_trim already behaves on real audio).
- Only report the full-30s trained-attack numbers as the headline result if
  the full-30s RANDOM-noise control is clearly worse (lower recall/F1/AUROC)
  than the full-30s trained-attack condition -- otherwise the effect is
  probably just "a lot of loud noise", not a property of the optimisation.
""")