'''
generate_speaker_embeddings.py
Loads a trained SpeakerEncoder model from a checkpoint, 
generates an embedding for each sample in a given dataset,
and saves the embeddings and corresponding labels to disk
'''

import yaml, argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.speaker_encoder import SpeakerEncoder
from utils.audio_triplet_loader import AudioTripletLoader

def generate_speaker_embeddings(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained SpeakerEncoder model from a checkpoint
    checkpoint_path = config['checkpoint_path']
    output_path = config['speaker_embedding_path']

    model_config = config['model']
    speaker_encoder = SpeakerEncoder(
        mel_dim=model_config['mel_dim'],
        hidden_dim=model_config['hidden_dim'],
        attn_dim=model_config['attn_dim'],
        embedding_out_dim=model_config['embedding_out_dim'],
        N_prenet=model_config['N_prenet'],
        N_conv=model_config['N_conv']
    )

    speaker_encoder = speaker_encoder.to(device)

    speaker_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    speaker_encoder.eval()

    # Load and preprocess audio data
    audio_data = AudioTripletLoader(config)
    audio_data_loader = DataLoader(audio_data, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    speaker_embeddings = []
    speaker_labels = []
    
    # Generate a speaker embedding for each sample,
    # store the embedding and corresponding label in a list
    with torch.no_grad():
        for anchor, _, _, speaker_id in tqdm(audio_data_loader):
            anchor = anchor.to(device)
            embedding = speaker_encoder(anchor)
            speaker_embeddings.append(embedding.cpu())
            speaker_labels.append(speaker_id)
    
    # Save the embeddings and labels to disk
    speaker_embeddings_tensor = torch.cat(speaker_embeddings, dim=0)
    speaker_labels_tensor = torch.cat(speaker_labels, dim=0)

    torch.save({"speaker_embeddings": speaker_embeddings_tensor, "speaker_labels": speaker_labels_tensor}, output_path)
    print(f"Speaker Embeddings saved to {output_path}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Speaker Embeddings")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    generate_speaker_embeddings(config)