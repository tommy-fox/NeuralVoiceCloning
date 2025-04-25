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

    # Load audio data
    audio_data = AudioTripletLoader(config)
    audio_data_loader = DataLoader(audio_data, batch_size=1, shuffle=False, num_workers=2)

    # Store multiple embeddings per speaker for plotting
    all_embeddings = []
    all_labels = []

    # For utterance-level access
    utterance_to_embedding = {}

    with torch.no_grad():
        for anchor, _, _, anchor_id, utterance_id in tqdm(audio_data_loader):
            anchor = anchor.to(device)  # [1, 1, n_mels, n_frames]
            embedding = speaker_encoder(anchor).squeeze(0).cpu()  # [embedding_dim]

            # Store per-speaker (many embeddings)
            all_embeddings.append(embedding)
            all_labels.append(anchor_id[0])  # Keep as str

            # Store per-utterance
            anchor_id_utterance_id_key = f"{anchor_id[0]}_{utterance_id[0]}"
            utterance_to_embedding[anchor_id_utterance_id_key] = embedding

    # Save all per-speaker embeddings (for t-SNE, etc.)
    speaker_embeddings_tensor = torch.stack(all_embeddings)
    speaker_labels_list = all_labels 

    torch.save({
        "speaker_embeddings": speaker_embeddings_tensor,
        "speaker_labels": speaker_labels_list
    }, output_path_base)

    print(f"All speaker embeddings saved to {output_path_base}")

    # Save utterance-level embeddings for training, keys are anchor id + utterance id
    torch.save(utterance_to_embedding, utterance_level_path)
    print(f"Utterance-level embeddings saved to {utterance_level_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Speaker Embeddings")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    generate_speaker_embeddings(config)
