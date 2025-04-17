import os
import torchaudio
import pandas as pd
from datasets import load_dataset
import torch
# Define all speaker configs
configs = [
    'irish_male', 'midlands_female', 'midlands_male',
    'northern_female', 'northern_male', 'scottish_female',
    'scottish_male', 'southern_female', 'southern_male',
    'welsh_female', 'welsh_male'
]

# Target directory to save audio files
output_dir = "./local_english_dialects"
os.makedirs(output_dir, exist_ok=True)

for config in configs:
    print(f"Downloading and saving config: {config}")
    dataset = load_dataset("ylacombe/english_dialects", config, split="train")

    speaker_dir = os.path.join(output_dir, config)
    os.makedirs(speaker_dir, exist_ok=True)

    metadata = []

    for i, sample in enumerate(dataset):
        wav_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        line_id = sample["line_id"]
        text = sample["text"]
        speaker_id = sample["speaker_id"]

        filename = f"{line_id}_{speaker_id}.wav"
        filepath = os.path.join(speaker_dir, filename)

        torchaudio.save(filepath, torchaudio.functional.resample(
            torch.tensor(wav_array).unsqueeze(0), orig_freq=sr, new_freq=16000
        ), sample_rate=16000)

        metadata.append({
            "filename": filename,
            "text": text,
            "speaker_id": speaker_id,
        })

    pd.DataFrame(metadata).to_csv(os.path.join(speaker_dir, "metadata.csv"), index=False)
    print(f"Saved {len(metadata)} files to {speaker_dir}")