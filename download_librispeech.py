# download_dataset.py
import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# Disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def download_dataset():
    print("=" * 60)
    print("DOWNLOADING SPEECH DATASET")
    print("=" * 60)
    
    # Create folder
    os.makedirs("audio_files", exist_ok=True)
    print("✅ Created folder: audio_files/")
    
    print("\n📥 Loading dataset...")
    
    # Try LibriSpeech first, fall back to Speech Commands
    try:
        print("   Trying LibriSpeech...")
        dataset = load_dataset(
            "librispeech_asr", 
            "clean", 
            split="train.clean.100", 
            streaming=True,
            trust_remote_code=True
        )
        print("   ✅ LibriSpeech loaded!")
    except:
        print("   ⚠️ LibriSpeech failed, using Speech Commands instead...")
        dataset = load_dataset(
            "speech_commands", 
            "v0.02", 
            split="validation",
            streaming=True,
            trust_remote_code=True
        )
        print("   ✅ Speech Commands loaded!")
    
    print("\n💾 Saving audio files...")
    count = 0
    max_samples = 50
    
    for sample in tqdm(dataset, total=max_samples):
        if count >= max_samples:
            break
            
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        
        filename = f"sample_{count:04d}.wav"
        filepath = os.path.join("audio_files", filename)
        
        sf.write(filepath, audio_array, sampling_rate)
        count += 1
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE!")
    print(f"   Total samples: {count}")
    print(f"   Location: audio_files/")
    print(f"   Sample rate: {sampling_rate}Hz")

if __name__ == "__main__":
    download_dataset()