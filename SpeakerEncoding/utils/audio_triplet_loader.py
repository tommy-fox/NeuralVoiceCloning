'''
audio_triplet_loader.py
This file parses speaker IDs from a dataset directory and maps each speaker ID to samples from that speaker.
When retrieving samples during testing, the class returns 3 total samples:
2 samples from the target speaker (anchor, positive) and 1 sample from a negative speaker.
The samples are drawn randomly from the dataset and converted into a Mel Spectrogram before being returned.
'''

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioTripletLoader(Dataset):
    def __init__(self, config):
        self.data_path = config["preprocessed_dataset_path"]
        self.sample_rate = config["sample_rate"]
        self.n_mels = config["n_mels"]
        self.n_fft = config["n_fft"]
        self.win_length = config["win_length"]
        self.hop_length = config["hop_length"]
        self.sample_duration = config.get("sample_duration", 1.0)  # in seconds
        self.max_frames = int(self.sample_duration * self.sample_rate)

        # Get speaker IDs from data directory, assumes sub directories are speaker ID's
        data_dir_contents = os.listdir(self.data_path)
        self.speakers = [
            speaker_id for speaker_id in data_dir_contents
            if os.path.isdir(os.path.join(self.data_path, speaker_id))
        ]

        # Map speaker ID to a list of samples for that speaker using files in data directory's subdirectories
        self.speaker_to_samples = {}
        for speaker in self.speakers:
            samples_for_speaker = [
                os.path.join(self.data_path, speaker, file)
                for file in os.listdir(os.path.join(self.data_path, speaker))
                if file.endswith(config["data_file_extension"])
            ]
            if len(samples_for_speaker) >= 2:
                self.speaker_to_samples[speaker] = samples_for_speaker

        self.speakers = list(self.speaker_to_samples.keys())

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        anchor_speaker = random.choice(self.speakers)

        negative_speaker = random.choice([
            speaker for speaker in self.speakers 
            if speaker != anchor_speaker
        ])

        anchor_sample, positive_sample = random.sample(self.speaker_to_samples[anchor_speaker], 2)
        negative_sample = random.choice(self.speaker_to_samples[negative_speaker])

        anchor_mel = self.load_mel_from_audio_file(anchor_sample)
        positive_mel = self.load_mel_from_audio_file(positive_sample)
        negative_mel = self.load_mel_from_audio_file(negative_sample)

        return anchor_mel, positive_mel, negative_mel

    def load_mel_from_audio_file(self, filepath):
        audio_data, audio_sample_rate = torchaudio.load(filepath)
        # Force all audio samples to be of the same duration
        if audio_data.shape[1] > self.max_frames:
            audio_data = audio_data[:, :self.max_frames]
        elif audio_data.shape[1] < self.max_frames:
            padding = self.max_frames - audio_data.shape[1]
            audio_data = torch.nn.functional.pad(audio_data, (0, padding))
        
        audio_data_mels = self.mel_transform(audio_data).squeeze(0)  #[n_mels, n_time_steps]

        # Normalize mel data (subtract mean, divide by standard deviation)
        audio_data_mels = (audio_data_mels - audio_data_mels.mean()) / (audio_data_mels.std())

        return audio_data_mels.unsqueeze(0)  #[1, n_mels, n_time_steps]
