# Universal Superposition Attack on Whisper ASR

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📜 Overview

This repository implements a powerful **universal adversarial attack** against OpenAI's **Whisper** Automatic Speech Recognition (ASR) model. By learning a single noise pattern, we can **mute** Whisper—forcing it to output an empty transcription—with a **100% success rate**.

Unlike the original "Muting Whisper" paper (Raina et al.), which uses a **prepend** attack (adding a 0.64-second clip to the beginning), our method uses a **global superposition** attack. The adversarial noise is mixed directly **on top** of the entire speech waveform, achieving perfect silencing across all tested audio files.

### Key Highlights
- ✅ **100% Success Rate** – Mutes Whisper on every tested audio file.
- 🧠 **Advanced Optimization** – Uses Momentum Iterative Fast Gradient Sign Method (MI-FGSM) for faster convergence.
- 🔇 **Imperceptible** – The noise is clamped to a strict amplitude limit (`0.05`) to remain quiet and inaudible.
- 💻 **CPU Compatible** – Runs efficiently even without a GPU.

---

## 🧪 How It Works

1. **Random Initialization** – Starts with a small random noise tensor (same length as the audio).
2. **Feedback Loop** – The noise is added to a batch of clean audio and passed through Whisper.
3. **Loss Calculation** – Computes the negative log-probability of the `<|nospeech|>` token.
4. **Gradient Update** – Uses MI-FGSM to nudge the noise in the direction that maximizes "silence" confidence.
5. **Constraint Enforcement** – Clamps the noise to the amplitude limit after every step.
6. **Convergence** – After just 25 steps, the noise becomes a universal "mute button" for Whisper.

---

## 📂 Project Structure
