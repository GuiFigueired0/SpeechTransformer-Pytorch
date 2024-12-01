import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datafeeder import DataFeeder
from model import SpeechTransformer, create_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
CLIP_GRAD_NORM = 1.0
MODEL_SAVE_PATH = "speech_transformer.pth"

# LOSS FUNCTION
def label_smoothing_loss(predictions, targets, vocab_size, smoothing=0.1):
    """
    Implements label smoothing loss.
    """
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (vocab_size - 1)
    one_hot = torch.zeros_like(predictions).scatter(1, targets.unsqueeze(1), confidence)
    one_hot = one_hot + smoothing_value
    log_probs = nn.functional.log_softmax(predictions, dim=-1)
    loss = -(one_hot * log_probs).sum(dim=-1).mean()
    return loss

# TRAINING FUNCTION
def train_one_epoch(model, datafeeder, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    train_data = datafeeder.get_batch()

    start_time = time.time()
    for step in range(len(datafeeder)//BATCH_SIZE):
        batch = next(train_data)
        
        inp = torch.tensor(batch['the_inputs'], dtype=torch.float32).to(DEVICE)  # Convert to float32 tensor. Shape: (batch, time, features)
        tar = torch.tensor(batch['the_labels'], dtype=torch.int64).to(DEVICE)   # Convert to int64 tensor. Shape: (batch, time)
        gtruth = torch.tensor(batch['ground_truth'], dtype=torch.int64).to(DEVICE)  # Convert to int64 tensor. Shape: (batch, time)

        tar_inp = tar[:, :-1]  # Input to decoder
        tar_real = gtruth[:, 1:]  # Targets for loss

        optimizer.zero_grad()
        
        # Masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        # Forward pass
        predictions = model(inp, tar_inp, training=True,
                            enc_padding_mask=enc_padding_mask,
                            look_ahead_mask=combined_mask,
                            dec_padding_mask=dec_padding_mask)
        loss = criterion(predictions.view(-1, predictions.size(-1)), tar_real.reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        acc = (predictions.argmax(dim=-1) == tar_real).float().mean().item()
        epoch_acc += acc

        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    avg_loss = epoch_loss / len(datafeeder)
    avg_acc = epoch_acc / len(datafeeder)
    print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f}s")
    print(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

def main():
    torch.manual_seed(42)
    
    print("Loading data...")
    train_feeder = DataFeeder(mode='dev', shuffle_data=True)

    print("Initializing model...")
    model = SpeechTransformer(target_vocab_size=len(train_feeder.vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = label_smoothing_loss

    print("Training model...")
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_feeder, optimizer, criterion, epoch)

    print("Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
