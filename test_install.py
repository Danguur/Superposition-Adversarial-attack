# test_install.py
import torch
import whisper
import numpy as np
import librosa

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Librosa version: {librosa.__version__}")

# Test GPU availability
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ️ Running on CPU (this is fine for testing)")

# Test Whisper
print("Testing Whisper...")
model = whisper.load_model("tiny")
print("✅ Whisper loaded successfully!")