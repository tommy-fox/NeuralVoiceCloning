'''
preprocess_audio.py
Resamples audio data and generates mel spectrograms
then saves the spectrograms to disk for later use
'''

import os
import yaml
import argparse
import torchaudio
import torch
from torchaudio.functional import resample

def preprocess_audio(
        input_dir,
        output_dir,
        target_sample_rate,
        audio_file_extension,
        mel_params
):
    os.makedirs(output_dir, exist_ok=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=mel_params['n_fft'],
        win_length=mel_params['win_length'],
        hop_length=mel_params['hop_length'],
        n_mels=mel_params['n_mels']
    )

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for filename in files:
            if not filename.lower().endswith(audio_file_extension):
                continue

            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_subdir, filename.replace(audio_file_extension, ".pt"))

            if os.path.exists(output_path):
                continue

            audio_data, original_sample_rate = torchaudio.load(input_path)
            if original_sample_rate != target_sample_rate:
                audio_data = resample(audio_data, orig_freq=original_sample_rate, new_freq=target_sample_rate)

            # Force all audio samples to be of the same length, so downstream comparisons are more consistent
            max_frames = int(mel_params.get("sample_duration", 1.0) * target_sample_rate)
            if audio_data.shape[1] > max_frames:
                audio_data = audio_data[:, :max_frames]
            elif audio_data.shape[1] < max_frames:
                padding = max_frames - audio_data.shape[1]
                audio_data = torch.nn.functional.pad(audio_data, (0, padding))

            # Standardize mel representation before saving
            mel = mel_transform(audio_data).squeeze(0)
            mel = (mel - mel.mean()) / mel.std()
            torch.save(mel.unsqueeze(0), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample and preprocess audio into mel spectrograms")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    input_dir = config['data']['dataset_path']
    output_dir = config['data']['preprocessed_dataset_path']
    target_sample_rate = config['data']['sample_rate']
    audio_file_extension = config['data']['data_file_extension']
    mel_params = {
        'n_mels': config['data']['n_mels'],
        'n_fft': config['data']['n_fft'],
        'win_length': config['data']['win_length'],
        'hop_length': config['data']['hop_length'],
        'sample_duration': config['data'].get('sample_duration', 1.0),
    }

    preprocess_audio(input_dir, output_dir, target_sample_rate, audio_file_extension, mel_params)
