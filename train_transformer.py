import os
import time
import torch
import torch.optim as optim

from datafeeder import DataFeeder, BATCH_SIZE
from model import SpeechTransformer, create_combined_mask, LabelSmoothingLoss

EPOCHS = 1
LAST_RUN = 0 # 0 for new training the model from scratch
CLIP_GRAD_NORM = 1.0
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, datafeeder, optimizer, criterion, epoch):
    model.train()
    learning_curve_path = os.path.join(os.getcwd(), 'model_weights', f"learning_curve.csv")
    train_data = datafeeder.get_batch()

    start_time = time.time()
    for step in range(len(datafeeder) // BATCH_SIZE):
        batch_time = time.time()
        batch = next(train_data)

        inp = torch.tensor(batch['the_inputs'], dtype=torch.float32).to(DEVICE)
        tar = torch.tensor(batch['the_labels'], dtype=torch.int64).to(DEVICE)
        gtruth = torch.tensor(batch['ground_truth'], dtype=torch.int64).to(DEVICE)

        optimizer.zero_grad()

        combined_mask = create_combined_mask(tar)
        predictions = model(
            inp,
            tar,
            enc_padding_mask=None,
            look_ahead_mask=combined_mask,
            dec_padding_mask=None,
        )
        loss = criterion(gtruth, predictions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        acc = (predictions.argmax(dim=-1) == gtruth).float().mean().item()
        
        if step % 100 == 0:
            with open(learning_curve_path, 'a', encoding='utf8') as file:
                file.write(f"\n{loss.item():.4f},{acc:.4f}")

        print(f"Epoch {epoch + 1}. Batch {step + 1}. Loss: {loss.item():.4f}. Accuracy: {acc:.4f}. Time: {time.time() - batch_time:.2f}s.")
    print(f'Total time of the epoch: {time.time() - start_time:.2f}s,')

def main():
    print("Loading data...")
    train_feeder = DataFeeder(mode='dev', shuffle_data=True)
    
    print("Initializing model...")
    model = SpeechTransformer(target_vocab_size=train_feeder.vocab_size()).to(DEVICE)
    criterion = LabelSmoothingLoss(vocab_size=train_feeder.vocab_size())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load partially trained model
    if LAST_RUN > 0:
        model_save_path = os.path.join(os.getcwd(), 'model_weights', f"speech_transformer_epoch{LAST_RUN}.pth")
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
    else:
        learning_curve_path = os.path.join(os.getcwd(), 'model_weights', f"learning_curve.csv")
        with open(learning_curve_path, 'w', encoding='utf8') as file:
            file.write(f"loss,accuracy")
            
    print("Starting training...")
    print(f"Number of steps by epoch: {len(train_feeder) // BATCH_SIZE}")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_one_epoch(model, train_feeder, optimizer, criterion, epoch)
        
        model_save_path = os.path.join(os.getcwd(), 'model_weights', f"speech_transformer_epoch{epoch+1+LAST_RUN}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
