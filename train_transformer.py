import os
import time
import torch
import torch.optim as optim

from datafeeder import DataFeeder, BATCH_SIZE, id2char
from model import SpeechTransformer, LabelSmoothingLoss, CustomSchedule, evaluate, create_masks

EPOCHS = 3
LAST_RUN = 0 # 0 for new training the model from scratch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, datafeeder, optimizer, criterion, epoch, learning_rate_schedule, step_count):
    model.train()
    learning_curve_path = os.path.join(os.getcwd(), 'extra', f"learning_curve.csv")
    train_data = datafeeder.get_batch()

    start_time = time.time()
    for step in range(len(datafeeder) // BATCH_SIZE):
        batch_time = time.time()
        batch = next(train_data)

        inp = torch.tensor(batch['the_inputs'], dtype=torch.float32).to(DEVICE)
        tar = torch.tensor(batch['the_labels'], dtype=torch.int64).to(DEVICE)
        gtruth = torch.tensor(batch['ground_truth'], dtype=torch.int64).to(DEVICE)

        # Adjust learning rate
        new_lr = learning_rate_schedule(step_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        optimizer.zero_grad()
        step_count += 1 

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
        predictions = model(
            inp,
            tar,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask,
        )
        loss = criterion(gtruth, predictions)
        loss.backward()
        optimizer.step()

        predictions = predictions.argmax(dim=-1)
        translated_gtruth = [ id2char(gtruth[i]) for i in range(gtruth.shape[0]) ]
        translated_predictions = [ id2char(predictions[i]) for i in range(predictions.shape[0]) ]
        cer, wer = evaluate(translated_predictions, translated_gtruth)
        acc = (predictions == gtruth).float().mean().item()
        
        if step % 100 == 0:
            with open(learning_curve_path, 'a', encoding='utf8') as file:
                file.write(f"\n{loss.item():.4f},{cer:.4f},{wer:.4f},{acc:.4f}")

        print(f"Epoch {epoch + 1}. Batch {step + 1}. Loss: {loss.item():.4f}. CER: {cer:.4f}. WER: {wer:.4f}. Acc: {acc:.4f}. Time: {time.time() - batch_time:.2f}s.")
    print(f'Total time of the epoch: {time.time() - start_time:.2f}s,')

def main():
    print("Loading data...")
    train_feeder = DataFeeder(mode='train', shuffle_data=True)
    steps_per_epoch = len(train_feeder) // BATCH_SIZE
    step_count = max(steps_per_epoch * LAST_RUN, 1)
    
    print("Initializing model...")
    model = SpeechTransformer(target_vocab_size=train_feeder.vocab_size()).to(DEVICE)
    learning_rate_schedule = CustomSchedule()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate_schedule(step_count),
                                 betas=(0.9, 0.98),
                                 eps=1e-9 )
    criterion = LabelSmoothingLoss(vocab_size=train_feeder.vocab_size())
    
    # Load partially trained model
    if LAST_RUN > 0:
        model_save_path = os.path.join(os.getcwd(), 'model_weights', f"speech_transformer_epoch{LAST_RUN}.pth")
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
    else:
        learning_curve_path = os.path.join(os.getcwd(), 'extra', f"learning_curve.csv")
        with open(learning_curve_path, 'w', encoding='utf8') as file:
            file.write(f"loss,cer,wer,acc")
            
    print("Starting training...")
    print(f"Number of steps by epoch: {steps_per_epoch}")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_one_epoch(model, train_feeder, optimizer, criterion, epoch, learning_rate_schedule, step_count)
        
        model_save_path = os.path.join(os.getcwd(), 'model_weights', f"speech_transformer_epoch{epoch+1+LAST_RUN}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
