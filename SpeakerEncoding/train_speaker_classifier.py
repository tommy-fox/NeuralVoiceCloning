'''
train_speaker_classifier.py
Trains a simple neural network to classify 
speaker IDs given their speaker embeddings.
This is intended to assess the performance of the speaker embedding model,
where the classifier will show high accuracy when the speaker embeddings
are well-defined for each speaker
'''
import yaml, argparse, time, torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from models.speaker_classifier import SpeakerEmbeddingClassifier

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load speaker embeddings and labels from file
    data = torch.load(config['speaker_embedding_path'])
    speaker_embeddings, speaker_labels = data['speaker_embeddings'], data['speaker_labels']
    dataset = TensorDataset(speaker_embeddings, speaker_labels)

    # Split data into train/validation/test
    model_config = config['speaker_classifier_model']
    num_samples = len(speaker_embeddings)
    n_train = int(model_config['train_split'] * num_samples)
    n_val = int(model_config['val_split'] * num_samples)
    n_test = num_samples - n_train - n_val

    train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    # Load train/validation/test data into DataLoaders
    train_loader = DataLoader(train_data, batch_size=model_config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=model_config['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=model_config['batch_size'], shuffle=False, num_workers=2)

    # Set up classifier model, loss, and optimizer
    speaker_embedding_dim = config['speaker_encoder_model']['embedding_out_dim']
    num_speakers = len(torch.unique(speaker_labels))
    speaker_classifier = SpeakerEmbeddingClassifier(embedding_dim=speaker_embedding_dim, classifier_dim=128, num_speakers=num_speakers)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(speaker_classifier.parameters(), lr=model_config['lr'])

    # Track best validation score
    best_val_acc = float("inf")

    for epoch in range(model_config['epochs']):
        print(f"\nEpoch {epoch}/{model_config['epochs']}")
        start = time.time()

        # Training
        speaker_classifier.train()
        total_pred, correct_pred, train_loss = 0.0, 0.0, 0.0
        for train_batch in tqdm(train_loader):
            speaker_embeddings, speaker_labels = train_batch
            speaker_embeddings = speaker_embeddings.to(device)
            speaker_labels = speaker_labels.to(device)

            classifer_output = speaker_classifier(speaker_embeddings).squeeze(1)
            loss = loss_function(classifer_output, speaker_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predictions = torch.max(classifer_output, 1)
            correct_pred += (predictions == speaker_labels).sum().item()
            total_pred += speaker_labels.size(0)
        
        train_acc = correct_pred / total_pred

        # Validation
        speaker_classifier.eval()
        total_pred, correct_pred, val_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_batch in tqdm(val_loader):
                speaker_embeddings, speaker_labels = train_batch
                speaker_embeddings = speaker_embeddings.to(device)
                speaker_labels = speaker_labels.to(device)

                classifer_output = speaker_classifier(speaker_embeddings).squeeze(1)
                loss = loss_function(classifer_output, speaker_labels)

                val_loss += loss.item()
                _, predictions = torch.max(classifer_output, 1)
                correct_pred += (predictions == speaker_labels).sum().item()
                total_pred += speaker_labels.size(0)
        
        val_acc = correct_pred / total_pred

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-start:.2f}s")

        if val_acc > best_val_acc:
            print("Best validation accuracy so far, saving model...")
            best_val_acc = val_acc
            torch.save(speaker_classifier.state_dict(), f"{config['checkpoint_save_dir']}/speaker_classifier_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    train(config)