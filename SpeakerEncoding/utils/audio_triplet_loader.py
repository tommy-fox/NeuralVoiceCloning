'''
audio_triplet_loader.py
Parses speaker IDs from a dataset directory and maps each speaker ID to samples from that speaker.
When retrieving samples during testing, the class returns 3 total samples:
2 samples from the target speaker (anchor, positive) and 1 sample from a negative speaker.
The samples are drawn randomly from the dataset and loaded as precomputed Mel Spectrogram tensors.
'''

import os
import random
import torch
from torch.utils.data import Dataset

class AudioTripletLoader(Dataset):
    def __init__(self, config):
        data_path = config['data']['preprocessed_dataset_path']

        # Get speaker IDs from data directory, assumes subdirectories are speaker IDs
        data_dir_contents = os.listdir(data_path)
        self.speakers = [
            speaker_id for speaker_id in data_dir_contents
            if os.path.isdir(os.path.join(data_path, speaker_id))
        ]

        # Map speaker ID to a list of mel spectrogram samples for that speaker
        self.speaker_to_samples = {}
        for speaker in self.speakers:
            samples_for_speaker = [
                os.path.join(data_path, speaker, file)
                for file in os.listdir(os.path.join(data_path, speaker))
                if file.endswith(".pt")
            ]
            if len(samples_for_speaker) >= 2:
                self.speaker_to_samples[speaker] = samples_for_speaker

        self.speakers = list(self.speaker_to_samples.keys())

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        anchor_id = random.choice(self.speakers)

        negative_speaker = random.choice([
            speaker for speaker in self.speakers 
            if speaker != anchor_id
        ])

        anchor_sample, positive_sample = random.sample(self.speaker_to_samples[anchor_id], 2)
        negative_sample = random.choice(self.speaker_to_samples[negative_speaker])

        anchor_mel = self.load_mel_from_file(anchor_sample)
        positive_mel = self.load_mel_from_file(positive_sample)
        negative_mel = self.load_mel_from_file(negative_sample)

        utterance_id = self.get_utterance_id(anchor_sample)
    
        return anchor_mel, positive_mel, negative_mel, anchor_id, utterance_id
    
    def get_utterance_id(self, path):
        # e.g., "p225/p225_001.pt" -> "p225_001"
        parts = os.path.normpath(path).split(os.sep)
        speaker = parts[-2]
        utterance = os.path.splitext(parts[-1])[0]

        if utterance.startswith(speaker + "_"):
            return utterance  # Already includes speaker prefix
        else:
            return f"{speaker}_{utterance}"

    def load_mel_from_file(self, filepath):
        mel_tensor = torch.load(filepath)  # Assumes [1, n_mels, n_frames] shape
        return mel_tensor