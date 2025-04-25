import yaml, argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def plot_tsne(speaker_embeddings, speaker_labels):
    speaker_embeddings = speaker_embeddings.squeeze(1)
    speaker_embeddings_numpy = speaker_embeddings.numpy()
    speaker_labels_numpy = np.array(speaker_labels)
    speaker_labels_encoded = LabelEncoder().fit_transform(speaker_labels_numpy)

    tsne = TSNE(random_state=42)
    speaker_embeddings_2d = tsne.fit_transform(speaker_embeddings_numpy)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(speaker_embeddings_2d[:, 0], speaker_embeddings_2d[:, 1], c=speaker_labels_encoded, cmap='tab20', alpha=1.0)
    plt.colorbar(scatter, label="Speaker ID")
    plt.title("t-SNE Plot of Speaker Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_speaker_embeddings_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    embedding_data = torch.load(config['speaker_embedding_path'])
    speaker_embeddings, speaker_labels = embedding_data['speaker_embeddings'], embedding_data['speaker_labels']
    plot_tsne(speaker_embeddings, speaker_labels)
