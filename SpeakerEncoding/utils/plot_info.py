import yaml, argparse
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pandas as pd
import matplotlib.colors as mcolors

def load_embeddings(path):
    data = torch.load(path)
    embeddings = data['speaker_embeddings'].squeeze(1)  # Shape: [N, D]
    labels = data['speaker_labels']                     # List[str]
    return embeddings, labels

def load_metadata(path):
    df = pd.read_csv(path)
    df = df.set_index("ID")
    return df

def compute_mean_embeddings(embeddings, labels):
    grouped = defaultdict(list)
    for emb, label in zip(embeddings, labels):
        grouped[label].append(emb)

    mean_embeddings = []
    speaker_ids = []
    for speaker_id, embs in grouped.items():
        stacked = torch.stack(embs)
        mean = stacked.mean(dim=0)
        mean_embeddings.append(mean)
        speaker_ids.append(speaker_id)

    return torch.stack(mean_embeddings), speaker_ids

def map_accent(accent):
    accent = accent.strip().lower()
    if accent in ["british", "english", "scottish", "welsh", "northernirish"]:
        return "Great Britain"
    elif accent == "irish":
        return "Ireland"
    elif accent in ["american", "canadian"]:
        return "North American"
    elif accent in ["australian english", "newzealand english", "southafrican"]:
        return "Southern Hemisphere"
    elif accent == "indian":
        return "Asia"
    else:
        return "Other"

def get_labels_from_metadata(speaker_ids, metadata_df, column, apply_accent_map=False):
    labels = []
    for sid in speaker_ids:
        if sid in metadata_df.index:
            raw = metadata_df.loc[sid][column]
            if apply_accent_map:
                labels.append(map_accent(raw))
            else:
                labels.append(raw.strip().capitalize())
        else:
            labels.append("Unknown")
    return labels

def plot_pca(embeddings, labels, label_type, save_path, colormap):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=encoded_labels, cmap=colormap, alpha=1.0)

    legend_labels = list(label_encoder.classes_)
    handles, _ = scatter.legend_elements()
    if len(handles) == len(legend_labels):
        plt.legend(handles=handles, labels=legend_labels, title=label_type)

    plt.title(f"PCA of Mean Speaker Embeddings by {label_type}")
    plt.xlabel("Pincipal Component 1")
    plt.ylabel("Pincipal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    embedding_path = "/Users/tom/Documents/GA Tech/SP 25 Deep Learning/Final Project/NeuralVoiceCloning/SpeakerEncoding/speaker_embeddings/generated_speaker_embeddings_epoch814.pt"
    metadata_path = "/Users/tom/Documents/GA Tech/SP 25 Deep Learning/Final Project/NeuralVoiceCloning/SpeakerEncoding/vctk_data/speaker-info.csv"

    embeddings, labels = load_embeddings(embedding_path)
    metadata_df = load_metadata(metadata_path)

    mean_embeddings, speaker_ids = compute_mean_embeddings(embeddings, labels)

    genders = get_labels_from_metadata(speaker_ids, metadata_df, "GENDER")
    accent_groups = get_labels_from_metadata(speaker_ids, metadata_df, "ACCENTS", apply_accent_map=True)

    plot_pca(mean_embeddings, genders, "Gender", "pca_by_gender.png", colormap="jet")
    plot_pca(mean_embeddings, accent_groups, "Accent Group", "pca_by_accent_group.png", colormap="jet")

if __name__ == "__main__":
    main()
