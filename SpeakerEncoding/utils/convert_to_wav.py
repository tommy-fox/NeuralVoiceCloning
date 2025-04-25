import os
import torchaudio
import torchaudio.transforms as T
import soundfile as sf

# Paths
input_dir = "NeuralVoiceCloning/StyleTTS2/vctk_data/wav24_flac/"
output_dir = "NeuralVoiceCloning/StyleTTS2/vctk_data/wav24/"

target_sr = 24000  # Target sample rate (to match StyleTTS's expected input)
os.makedirs(output_dir, exist_ok=True)

def convert_to_wav(input_path, output_path, target_sr=24000):
    waveform, sample_rate = torchaudio.load(input_path)
    
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Save as WAV
    sf.write(output_path, waveform.squeeze(0).numpy(), target_sr)

# Walk through all files
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith((".flac", ".wav", ".mp3", ".wv")):  # Expand this if needed
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            convert_to_wav(input_path, output_path)

print("Done converting all files")
