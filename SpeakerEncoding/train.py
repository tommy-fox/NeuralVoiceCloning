'''
train.py
Reads parameters from a configuration file to train a Speaker Encoder model
'''
import yaml, argparse, time, torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.audio_triplet_loader import AudioTripletLoader
from models.speaker_encoder import SpeakerEncoder
from utils.data_utils import AverageMeter

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audioDataTriplets = AudioTripletLoader(config)
    dataloader = DataLoader(audioDataTriplets, config["batch_size"], shuffle=True, num_workers=4)

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

    loss_function = nn.TripletMarginLoss(margin=config['triplet_loss_margin'])
    optimizer = optim.Adam(speaker_encoder.parameters(), lr=config['lr'])

    start_time = time.time()
    loss_meter = AverageMeter()
    iter_meter = AverageMeter()

    for epoch in range(config['epochs']):
        speaker_encoder.train()
        loss = 0.0
        for anchor_sample, positive_sample, negative_sample in tqdm(dataloader):
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

            loss += loss.item()

            loss_meter.update(loss.item())
            iter_meter.update(time.time()-start_time)
        
        print(
            f'Epoch: [{epoch}][{loss}/{len(dataloader):.3f}]\t'
            f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
            f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
            )
        
        torch.save(speaker_encoder.state_dict(), f"{config['save_dir']}/encoder_epoch{epoch}.pt")
    
    print(f"Completed in {(time.time()-start_time):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Encoder Trainer")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    train(config)


    