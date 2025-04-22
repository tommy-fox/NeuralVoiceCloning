'''
generate_speaker_embeddings.py
Loads a trained SpeakerEncoder model from a checkpoint, 
generates an embedding for each sample in a given dataset,
saves a map of the speaker_id to their average embedding for downstream plotting,
and saves a map of the speaker_id + utterance_id to the corresponding embedding
for downstream StyleTTS tuning
'''

import yaml, argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from models.speaker_encoder import SpeakerEncoder
from utils.audio_triplet_loader import AudioTripletLoader

def generate_speaker_embeddings(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained SpeakerEncoder model from a checkpoint
    checkpoint_path = config['checkpoint_path']
    output_path_base = config['speaker_embedding_path']
    utterance_level_path = output_path_base.replace(".pt", "_utterance.pt")

    model_config = config['speaker_encoder_model']
    speaker_encoder = SpeakerEncoder(
        mel_dim=model_config['mel_dim'],
        hidden_dim=model_config['hidden_dim'],
        attn_dim=model_config['attn_dim'],
        embedding_out_dim=model_config['embedding_out_dim'],
        N_prenet=model_config['N_prenet'],
        N_conv=model_config['N_conv']
    ).to(device)

    speaker_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    speaker_encoder.eval()

    # Load and preprocess audio data
    audio_data = AudioTripletLoader(config)
    audio_data_loader = DataLoader(audio_data, batch_size=1, shuffle=False, num_workers=2)

    # For speaker-level mean
    speaker_to_embeddings = defaultdict(list)

    # For utterance-level
    utterance_to_embedding = {}

    with torch.no_grad():
        for anchor, _, _, anchor_id, utterance_id in tqdm(audio_data_loader):
            anchor = anchor.to(device)  # [1, 1, n_mels, n_frames]
            embedding = speaker_encoder(anchor).squeeze(0).cpu()  # [embedding_dim]

            # speaker-level (for plotting)
            speaker_to_embeddings[anchor_id[0]].append(embedding)

            # utterance-level (for TTS finetune)
            key = f"{anchor_id[0]}_{utterance_id[0]}"
            utterance_to_embedding[key] = embedding

    # Compute mean embeddings for each speaker
    speaker_mean_embeddings = []
    speaker_labels = []

    for speaker, embeddings in speaker_to_embeddings.items():
        stacked = torch.stack(embeddings)
        speaker_mean_embeddings.append(stacked.mean(dim=0))
        speaker_labels.append(speaker)

    # Save speaker-level file
    speaker_embeddings_tensor = torch.stack(speaker_mean_embeddings)
    speaker_labels_tensor = speaker_labels  # Keep as list of strings

    torch.save({
        "speaker_embeddings": speaker_embeddings_tensor,
        "speaker_labels": speaker_labels_tensor
    }, output_path_base)
    print(f"Speaker-level embeddings saved to {output_path_base}")

    # Save utterance-level file
    torch.save(utterance_to_embedding, utterance_level_path)
    print(f"Utterance-level embeddings saved to {utterance_level_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Speaker Embeddings")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    generate_speaker_embeddings(config)
