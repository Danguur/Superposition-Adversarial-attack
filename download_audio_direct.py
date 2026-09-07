# download_audio_direct.py
import os
import requests
import soundfile as sf
import numpy as np
from tqdm import tqdm

def download_audio_direct():
    """
    Download audio files directly from public URLs
    No Hugging Face needed!
    """
    print("=" * 60)
    print("DOWNLOADING AUDIO FILES (DIRECT METHOD)")
    print("=" * 60)
    
    # Create folder
    os.makedirs("audio_files", exist_ok=True)
    print("✅ Created folder: audio_files/")
    
    # Public domain audio URLs (from Open Speech Repository)
    # These are 8kHz, we'll resample to 16kHz
    audio_urls = [
        # American English sample sentences
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0011_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0012_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0013_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0014_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0015_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0016_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0017_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0018_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0019_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0020_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0021_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0022_8k.wav",
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0023_8k.wav",
    ]
    
    print(f"\n📥 Downloading {len(audio_urls)} audio files...")
    
    for i, url in enumerate(tqdm(audio_urls, desc="Downloading")):
        try:
            # Download the file
            response = requests.get(url, timeout=30)
            
            # Save the raw WAV file
            filename = f"sample_{i+1:04d}.wav"
            filepath = os.path.join("audio_files", filename)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 60)
    
    # Count files
    audio_files = [f for f in os.listdir("audio_files") if f.endswith('.wav')]
    print(f"   Total samples: {len(audio_files)}")
    print(f"   Location: audio_files/")
    print(f"   Sample rate: 8kHz (Whisper will handle this)")
    
    # Verify one file
    if audio_files:
        import librosa
        sample_path = os.path.join("audio_files", audio_files[0])
        y, sr = librosa.load(sample_path, sr=None)
        print(f"\n📄 Sample file: {audio_files[0]}")
        print(f"   Duration: {len(y)/sr:.2f}s")
        print(f"   Sample rate: {sr}Hz")

if __name__ == "__main__":
    download_audio_direct()