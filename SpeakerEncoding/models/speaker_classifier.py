import torch.nn as nn
'''
speaker_classifier.py
Implements a simple classification neural network
intended to classify speakers based on speaker embedding
'''


class SpeakerEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, classifier_dim, num_speakers):
        super().__init__()
        self.speaker_classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_dim),
            nn.ReLU(),
            nn.Linear(classifier_dim, num_speakers)
        )

    def forward(self, x):
        return self.speaker_classifier(x)