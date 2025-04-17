import random
import torch
import torchaudio
from torch.utils.data import Dataset

class HuggingFaceTripletLoader(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.sample_rate = config["sample_rate"]
        self.n_mels = config["n_mels"]
        self.n_fft = config["n_fft"]
        self.win_length = config["win_length"]
        self.hop_length = config["hop_length"]
        self.sample_duration = config.get("sample_duration", 1.0)
        self.max_frames = int(self.sample_duration * self.sample_rate)

        self.speaker_to_indices = {}
        for idx, sample in enumerate(self.dataset):
            speaker = sample["speaker_label"]
            if speaker not in self.speaker_to_indices:
                self.speaker_to_indices[speaker] = []
            self.speaker_to_indices[speaker].append(idx)

        self.speakers = list(self.speaker_to_indices.keys())

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def __len__(self):
        return 100000  # or adjust based on len(self.dataset)

    def __getitem__(self, idx):
        anchor_speaker = random.choice(self.speakers)
        negative_speaker = random.choice([s for s in self.speakers if s != anchor_speaker])

        anchor_idx, positive_idx = random.sample(self.speaker_to_indices[anchor_speaker], 2)
        negative_idx = random.choice(self.speaker_to_indices[negative_speaker])

        anchor_mel = self._load_mel(anchor_idx)
        positive_mel = self._load_mel(positive_idx)
        negative_mel = self._load_mel(negative_idx)

        return anchor_mel, positive_mel, negative_mel

    def _load_mel(self, idx):
        audio_obj = self.dataset[idx]["audio"]
        audio_array = torch.tensor(audio_obj["array"]).unsqueeze(0).float()

        if audio_array.shape[1] > self.max_frames:
            audio_array = audio_array[:, :self.max_frames]
        elif audio_array.shape[1] < self.max_frames:
            padding = self.max_frames - audio_array.shape[1]
            audio_array = torch.nn.functional.pad(audio_array, (0, padding))

        mel = self.mel_transform(audio_array).squeeze(0)
        mel = (mel - mel.mean()) / mel.std()
        return mel.unsqueeze(0)