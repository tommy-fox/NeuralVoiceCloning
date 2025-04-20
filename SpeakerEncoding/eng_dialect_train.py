import os
import time
import yaml
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset, Audio
from torch.utils.data import DataLoader, random_split

from utils.h_f_triplet_loader import HuggingFaceTripletLoader
from utils.data_utils import AverageMeter
from models.speaker_encoder import SpeakerEncoder

def load_combined_dataset(config):
    local_dir = config["local_dataset_dir"]
    configs = config["huggingface_configs"]

    dataset_list = []
    for c in configs:
        subdir = os.path.join(local_dir, c)
        audio_files = [
            {"audio": os.path.join(subdir, f), "speaker_label": c}
            for f in os.listdir(subdir)
            if f.endswith(".wav")
        ]
        ds = Dataset.from_list(audio_files)
        ds = ds.cast_column("audio", Audio(sampling_rate=config["sample_rate"]))
        dataset_list.append(ds)

    return concatenate_datasets(dataset_list)

def load_dataset_splits(config):
    raw_dataset = load_combined_dataset(config)
    full_dataset = HuggingFaceTripletLoader(raw_dataset, config)
    total_num_samples = len(full_dataset)

    n_train = int(config['train_split'] * total_num_samples)
    n_val = int(config['val_split'] * total_num_samples)
    n_test = total_num_samples - n_train - n_val

    train_split, val_split, test_split = random_split(
        full_dataset,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = config['batch_size']

    train_data = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=2)
    test_data = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_data, val_data, test_data

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data = load_dataset_splits(config)

    model_config = config['model']

    speaker_encoder = SpeakerEncoder(
        mel_dim=model_config['mel_dim'],
        hidden_dim=model_config['hidden_dim'],
        attn_dim=model_config['attn_dim'],
        embedding_out_dim=model_config['embedding_out_dim'],
        N_prenet=model_config['N_prenet'],
        N_conv=model_config['N_conv']
    ).to(device)

    loss_function = nn.TripletMarginLoss(margin=config['triplet_loss_margin'])
    optimizer = optim.Adam(speaker_encoder.parameters(), lr=config['lr'])

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        start_time = time.time()

        speaker_encoder.train()
        train_loss_meter = AverageMeter()
        for anchor_sample, positive_sample, negative_sample in tqdm(train_data):
            anchor_sample = anchor_sample.to(device)
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)

            anchor_embedding = speaker_encoder(anchor_sample)
            positive_embedding = speaker_encoder(positive_sample)
            negative_embedding = speaker_encoder(negative_sample)

            loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        speaker_encoder.eval()
        val_loss_meter = AverageMeter()
        for anchor_sample, positive_sample, negative_sample in tqdm(val_data):
            anchor_sample = anchor_sample.to(device)
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)

            with torch.no_grad():
                anchor_embedding = speaker_encoder(anchor_sample)
                positive_embedding = speaker_encoder(positive_sample)
                negative_embedding = speaker_encoder(negative_sample)

                loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)
                val_loss_meter.update(loss.item())

        print(f"Train Loss: {train_loss_meter.avg:.6f} | Val Loss: {val_loss_meter.avg:.6f} | Time: {time.time() - start_time:.2f}s")

        if val_loss_meter.avg < best_val_loss:
            print("Best validation loss so far, saving model")
            best_val_loss = val_loss_meter.avg
            torch.save(speaker_encoder.state_dict(), f"{config['save_dir']}/encoder_epoch{epoch}.pt")

    print(f"Training completed in {(time.time()-start_time):.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Encoder Trainer")
    parser.add_argument("--config_file", type=str, default="configs/eng_dialect_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    train(config)

