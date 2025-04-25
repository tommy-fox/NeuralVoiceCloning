import torch
import torch.nn as nn
import torch.nn.functional as F

'''
speaker_classifier.py
Implements the speaker classification architecture from:
"Neural Voice Cloning with a Few Samples" (https://arxiv.org/pdf/1802.06006)
Takes speaker embeddings as input and classifies the speaker, 
intended to verify the speaker encoding model 
'''

class SpeakerEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_speakers=108):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
        )

        self.output_embedding = nn.Linear(256, 32)
        self.classifier = nn.Linear(32, num_speakers)

    def forward(self, x):
        """
        Input: x of shape [batch_size, embedding_dim]
        Output: logits of shape [batch_size, num_speakers]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv_layers(x)
        x = x.mean(dim=2)
        x = self.output_embedding(x)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits
