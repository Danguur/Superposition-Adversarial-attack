import os
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)



SAMPLE_RATE = 16000
SEGMENT_LENGTH_SEC = 0.64
NUM_NOISE_SAMPLES = int(SEGMENT_LENGTH_SEC * SAMPLE_RATE)
AMPLITUDE_LIMIT = 0.02
MODEL_NAME = "tiny.en"

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.01

AUDIO_FOLDER = "audio_files"
ATTACK_SAVE_PATH = "attack_segment.pt"
RESULTS_CSV = "muting_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_random_noise(duration_sec=SEGMENT_LENGTH_SEC, amplitude=AMPLITUDE_LIMIT,
                       sample_rate=SAMPLE_RATE, seed=None):
    rng = np.random.default_rng(seed)
    num_samples = int(duration_sec * sample_rate)
    return rng.uniform(-amplitude, amplitude, num_samples).astype(np.float32)



class MutingAttack:
    def __init__(self, model):
        self.model = model
        self.amplitude_limit = AMPLITUDE_LIMIT
        self.segment_length = SEGMENT_LENGTH_SEC
        self.sample_rate = SAMPLE_RATE
        self.num_samples = NUM_NOISE_SAMPLES

        self.delta = torch.randn(self.num_samples, device=device, requires_grad=True)
        self.delta.data *= 0.01

        self.tokenizer = whisper.tokenizer.get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages, task="transcribe"
        )
        
        self.no_speech_token = self.tokenizer.no_speech
        self.sot_sequence = list(self.tokenizer.sot_sequence_including_notimestamps)

        print("\n✅ Attack initialized!")
        print(f"   Noise length: {self.num_samples} samples ({self.segment_length}s)")
        print(f"   Amplitude limit: ±{self.amplitude_limit}")

    def prepare_audio(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        return torch.from_numpy(audio).float().to(device)

    def prepend_attack(self, audio, keep_grad=False):
        attack = torch.clamp(self.delta, -self.amplitude_limit, self.amplitude_limit)
        attacked = torch.cat([attack, audio])
        if keep_grad:
            attacked = whisper.pad_or_trim(attacked)
            return attacked
        else:
            attacked_np = attacked.detach().cpu().numpy()
            attacked_np = whisper.pad_or_trim(attacked_np)
            return torch.from_numpy(attacked_np).float().to(device)

 
    def get_no_speech_prob(self, audio_tensor, keep_grad=False):
        if keep_grad:
            mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=self.model.dims.n_mels)
            mel = mel.unsqueeze(0).to(device)
            encoder_output = self.model.encoder(mel)
            sot = torch.tensor([self.sot_sequence], device=device, dtype=torch.long)
            logits = self.model.decoder(sot, encoder_output)
            last_step_logits = logits[0, -1]
            log_probs = F.log_softmax(last_step_logits, dim=-1)
            return log_probs[self.no_speech_token].exp()
        else:
            audio_np = audio_tensor.detach().cpu().numpy()
            mel = whisper.log_mel_spectrogram(audio_np, n_mels=self.model.dims.n_mels)
            mel = mel.unsqueeze(0).to(device)
            with torch.no_grad():
                encoder_output = self.model.encoder(mel)
                sot = torch.tensor([self.sot_sequence], device=device, dtype=torch.long)
                logits = self.model.decoder(sot, encoder_output)
                last_step_logits = logits[0, -1]
                log_probs = F.log_softmax(last_step_logits, dim=-1)
            return log_probs[self.no_speech_token].exp()

    def compute_loss(self, audio_batch):
        total_loss = 0.0
        for audio in audio_batch:
            attacked = self.prepend_attack(audio, keep_grad=True)
            prob = self.get_no_speech_prob(attacked, keep_grad=True)
            loss = -torch.log(prob + 1e-8)
            total_loss += loss
        return total_loss / len(audio_batch)

    def train_attack(self, audio_files, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
        print("\n" + "=" * 60)
        print("TRAINING LEARNED ADVERSARIAL ATTACK (fixed loss target)")
        print("=" * 60)
        print(f"   Samples: {len(audio_files)}  Epochs: {epochs}  "
              f"Batch: {batch_size}  LR: {lr}")

        optimizer = torch.optim.Adam([self.delta], lr=lr)
        best_success = 0.0
        best_delta = None

        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(audio_files)
            num_batches = 0

            for i in range(0, len(audio_files), batch_size):
                batch_files = audio_files[i:i + batch_size]
                batch_audio = [self.prepare_audio(f) for f in batch_files]

                optimizer.zero_grad()
                loss = self.compute_loss(batch_audio)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.delta], 1.0)
                optimizer.step()

                with torch.no_grad():
                    self.delta.data = torch.clamp(
                        self.delta.data, -self.amplitude_limit, self.amplitude_limit
                    )
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)

            if epoch % 5 == 0 or epoch == epochs - 1:
                success = self.evaluate_attack(audio_files[:10])
                print(f"Epoch {epoch:03d}: Loss = {avg_loss:.4f}, "
                      f"Black-box success (transcribe()) = {success:.1f}%")
                if success > best_success:
                    best_success = success
                    best_delta = self.delta.detach().clone()
                    print(f"   🎯 New best: {best_success:.1f}%")

        if best_delta is not None:
            self.delta = best_delta
            self.delta.requires_grad = True

        print(f"\n✅ Training complete! Best black-box success: {best_success:.1f}%")
        return self.delta

    def evaluate_attack(self, audio_files, suppress_blank=True, without_timestamps=False):
        """
        Black-box evaluation via model.transcribe(), matching real deployment.
        suppress_blank / without_timestamps let us ablate Whisper's own
        blank-suppression defenses to test whether they are what's blocking
        the attack (see the diagnostic discussed before this fix).
        """
        success = 0
        total = len(audio_files)
        for audio_file in audio_files:
            audio = self.prepare_audio(audio_file)
            attacked = self.prepend_attack(audio)
            audio_np = attacked.detach().cpu().numpy()
            try:
                result = self.model.transcribe(
                    audio_np, suppress_blank=suppress_blank,
                    without_timestamps=without_timestamps,
                )
                text = result["text"].strip()
                if text == "":
                    success += 1
            except Exception:
                pass
        return (success / total) * 100 if total else 0.0

    def save_attack(self, path=ATTACK_SAVE_PATH):
        torch.save({
            "delta": self.delta.detach().cpu(),
            "amplitude_limit": self.amplitude_limit,
            "segment_length": self.segment_length,
            "num_samples": self.num_samples,
        }, path)
        print(f"✅ Attack saved to {path}")


# ============================================================
# EVALUATION HELPERS (shared style with the superposition script)
# ============================================================

def transcribes_empty(model, waveform_np, suppress_blank=True, without_timestamps=False):
    result = model.transcribe(waveform_np.astype(np.float32),
                               suppress_blank=suppress_blank,
                               without_timestamps=without_timestamps)
    return result["text"].strip() == ""


def get_real_no_speech_prob(model, waveform_np):
    """Whisper's own no_speech_prob, via whisper.decode (not the training proxy)."""
    mel = whisper.log_mel_spectrogram(waveform_np.astype(np.float32), n_mels=model.dims.n_mels)
    mel = mel.to(device)
    options = whisper.DecodingOptions(
        fp16=(device == "cuda"), without_timestamps=True, language="en"
    )
    result = whisper.decode(model, mel, options)
    if isinstance(result, list):
        result = result[0]
    return float(result.no_speech_prob)


def build_condition(model, file_paths, noise_1d=None, suppress_blank=True, without_timestamps=False):
    rows = []
    for f in file_paths:
        audio = whisper.load_audio(f)
        audio = whisper.pad_or_trim(audio)  # 30s, matches original prepare_audio() length
        if noise_1d is not None:
            attacked = np.concatenate([noise_1d, audio]).astype(np.float32)
            attacked = whisper.pad_or_trim(attacked)  # trim back to 30s, as in prepend_attack()
        else:
            attacked = audio.astype(np.float32)
        empty = transcribes_empty(model, attacked, suppress_blank, without_timestamps)
        nsp = get_real_no_speech_prob(model, attacked)
        rows.append({"file": os.path.basename(f), "muted": int(empty), "no_speech_prob": nsp})
    return rows


def paired_metrics(clean_rows, attacked_rows, label):
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
        auroc = float("nan")

    print(f"\n--- {label} ---")
    print("Confusion matrix (rows=true [0=clean,1=attacked], cols=pred [0=not muted,1=muted]):")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["not_muted_true", "muted_true"],
                                 zero_division=0))
    print(f"AUROC (no_speech_prob, clean vs attacked): {auroc:.4f}")

    return {"label": label, "confusion_matrix": cm.tolist(),
            "precision_class1": precision[1], "recall_class1": recall[1],
            "f1_class1": f1[1], "auroc": auroc}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MUTING ATTACK v2 - fixed loss target + full evaluation")
    print("=" * 60)

    print("\n📥 Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME).to(device)
    print(f"   ✅ Model loaded on: {device}")

    if not os.path.exists(AUDIO_FOLDER):
        print(f"❌ Folder '{AUDIO_FOLDER}' not found!")
        return
    audio_files = [os.path.join(AUDIO_FOLDER, f) for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")]
    print(f"   Found {len(audio_files)} audio files")
    if len(audio_files) < 5:
        print("❌ Not enough audio files (need at least 5)")
        return

    print("\n📊 Attack Parameters:")
    print("   Attack Type: Prepend | Segment: "
          f"{SEGMENT_LENGTH_SEC}s ({NUM_NOISE_SAMPLES} samples) | "
          f"Amplitude: ±{AMPLITUDE_LIMIT} | Epochs: {EPOCHS}")

    # ---- DIAGNOSTIC: confirm the SuppressBlank / timestamp-masking hypothesis ----
    # Uses a fresh random (untrained) noise segment purely to check whether
    # transcribe()'s own defenses -- not the perturbation -- were blocking
    # the previous version's result.
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: is transcribe()'s blank-suppression blocking the attack?")
    print("=" * 60)
    diag_noise = make_random_noise(seed=0)
    diag_file = audio_files[0]
    audio = whisper.load_audio(diag_file)
    noisy = whisper.pad_or_trim(np.concatenate([diag_noise, audio]).astype(np.float32))
    default_text = model.transcribe(noisy)["text"]
    open_text = model.transcribe(noisy, suppress_blank=False, without_timestamps=True)["text"]
    print(f"   Default decode (suppress_blank=True):  '{default_text.strip()}'")
    print(f"   Defenses off   (suppress_blank=False):  '{open_text.strip()}'")
    print("   (If these differ, the earlier 0% result was at least partly caused by")
    print("    transcribe()'s own decode-time defenses, not perturbation weakness.)")

    # ---- Random-noise baseline (untrained), same style as original Step A ----
    print("\n" + "=" * 60)
    print("STEP A: RANDOM NOISE BASELINE (untrained)")
    print("=" * 60)
    baseline_noise = make_random_noise(seed=1)
    baseline_success = 0
    for f in audio_files[:10]:
        audio = whisper.load_audio(f)
        noisy = whisper.pad_or_trim(np.concatenate([baseline_noise, audio]).astype(np.float32))
        if transcribes_empty(model, noisy):
            baseline_success += 1
    print(f"Random noise baseline success rate: {baseline_success / 10 * 100:.1f}%")

    # ---- Train the fixed attack ----
    attacker = MutingAttack(model)
    attacker.train_attack(list(audio_files), epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
    attacker.save_attack(ATTACK_SAVE_PATH)
    trained_noise = attacker.delta.detach().cpu().numpy()

    # ---- Final black-box evaluation (both with and without Whisper's own defenses) ----
    print("\n" + "=" * 60)
    print("FINAL BLACK-BOX EVALUATION (model.transcribe())")
    print("=" * 60)
    default_rate = attacker.evaluate_attack(audio_files, suppress_blank=True, without_timestamps=False)
    open_rate = attacker.evaluate_attack(audio_files, suppress_blank=False, without_timestamps=True)
    print(f"Trained attack success (default decode, defenses ON):  {default_rate:.1f}%")
    print(f"Trained attack success (defenses OFF, ablation):        {open_rate:.1f}%")

    # ---- Full confusion-matrix / precision / recall / F1 / AUROC evaluation ----
    print("\n" + "=" * 60)
    print("FULL EVALUATION: confusion matrix, precision/recall/F1, AUROC")
    print("=" * 60)

    conditions = {
        "clean":                dict(noise=None,          suppress_blank=True,  without_timestamps=False),
        "random_noise":         dict(noise=baseline_noise, suppress_blank=True,  without_timestamps=False),
        "trained_noise":        dict(noise=trained_noise,  suppress_blank=True,  without_timestamps=False),
        "clean_defenses_off":   dict(noise=None,          suppress_blank=False, without_timestamps=True),
        "trained_defenses_off": dict(noise=trained_noise,  suppress_blank=False, without_timestamps=True),
    }
    results = {}
    for name, cfg in conditions.items():
        print(f"\nRunning condition: {name} ...")
        rows = build_condition(model, audio_files, cfg["noise"], cfg["suppress_blank"], cfg["without_timestamps"])
        results[name] = rows
        rate = 100.0 * sum(r["muted"] for r in rows) / len(rows)
        print(f"  -> muted rate: {rate:.1f}% ({sum(r['muted'] for r in rows)}/{len(rows)})")

    with open(RESULTS_CSV, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["condition", "file", "muted", "no_speech_prob"])
        for name, rows in results.items():
            for r in rows:
                writer.writerow([name, r["file"], r["muted"], f"{r['no_speech_prob']:.6f}"])
    print(f"\n✅ Per-sample results written to {RESULTS_CSV}")

    summary = []
    summary.append(paired_metrics(results["clean"], results["random_noise"],
                                   "Random noise vs clean (default decode, defenses ON)"))
    summary.append(paired_metrics(results["clean"], results["trained_noise"],
                                   "Trained attack vs clean (default decode, defenses ON)"))
    summary.append(paired_metrics(results["clean_defenses_off"], results["trained_defenses_off"],
                                   "Trained attack vs clean (defenses OFF ablation)"))

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Condition':55s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUROC':>7s}")
    for s in summary:
        print(f"{s['label']:55s} {s['precision_class1']:.3f}  {s['recall_class1']:.3f}  "
              f"{s['f1_class1']:.3f}  {s['auroc']:.3f}")

    print("""
INTERPRETATION GUIDE:
- Compare 'trained attack vs clean (defenses ON)' to 'trained attack vs clean
  (defenses OFF)'. If success/F1/recall jump substantially with defenses off,
  that confirms Whisper's own SuppressBlank / timestamp-forcing rules -- not
  perturbation weakness -- were the main blocker in the original script.
- Compare 'random noise vs clean' to 'trained noise vs clean' (both defenses
  ON). If trained clearly beats random, the optimisation is adding real value
  on top of just adding noise.
""")


if __name__ == "__main__":
    main()