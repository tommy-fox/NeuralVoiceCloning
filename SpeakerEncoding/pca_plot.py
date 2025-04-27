import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from collections import Counter

data = torch.load('speaker_embeddings/generated_speaker_embeddings.pt', map_location='cpu')
embeddings = data['speaker_embeddings'].cpu().numpy()
labels = data['speaker_labels']

if embeddings.ndim == 3 and embeddings.shape[1] == 1:
    embeddings = embeddings.squeeze(1)

print(f"Loaded {embeddings.shape[0]} embeddings with {len(set(labels))} unique speakers.")

group_counter = Counter()
for label in labels:
    if label.endswith('male'):
        group_counter['male'] += 1
    elif label.endswith('female'):
        group_counter['female'] += 1
    else:
        group_counter['unknown'] += 1

print("Group counts:", group_counter)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

group_labels = []
for label in labels:
    if label.endswith('male'):
        group_labels.append('male')
    elif label.endswith('female'):
        group_labels.append('female')
    else:
        group_labels.append('unknown')

group_to_color = {'male': 0, 'female': 1, 'unknown': 2}
colors = [group_to_color[g] for g in group_labels]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=colors,
    cmap=plt.get_cmap('coolwarm', 3),  #
    alpha=0.7,
    s=10
)

handles = [
    mpatches.Patch(color=plt.cm.coolwarm(0.0), label='Male'),
    mpatches.Patch(color=plt.cm.coolwarm(1.0), label='Female'),
]
plt.legend(
    handles=handles,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title="Group",
    fontsize='small'
)

plt.title('PCA of Speaker Embeddings (Male vs Female)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_speaker_gender_fixed.png', dpi=300)
plt.show()
