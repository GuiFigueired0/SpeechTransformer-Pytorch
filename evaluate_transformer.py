import os
import time
import torch

from datafeeder import DataFeeder, id2char, BATCH_SIZE, SOS, EOS
from model import SpeechTransformer, evaluate

LAST_RUN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, datafeeder):
    model.eval()
    test_data = datafeeder.get_batch()

    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0

    start_time = time.time()
    n_batches = len(datafeeder) // BATCH_SIZE
    print(f"Number of batches: {n_batches}")
    with torch.no_grad():
        for step in range(n_batches):
            batch = next(test_data)
            batch_time = time.time()
            inp = torch.tensor(batch['the_inputs'], dtype=torch.float32).to(DEVICE)
            gtruth = torch.tensor(batch['ground_truth'], dtype=torch.int64).to(DEVICE)

            translated_gtruth = [id2char(gtruth[i]) for i in range(gtruth.size(0))]
            sequence_sizes = [ len(seq) - 8 for seq in translated_gtruth ] # -8 is for the SOS> and EOS> excess
            
            decoded_sequences = model.beam_search_decoding(inp, sequence_sizes, SOS, EOS)
            translated_predictions = [id2char(seq.tolist()) for seq in decoded_sequences]

            cer, wer = evaluate(translated_predictions, translated_gtruth)
            total_cer += cer * len(batch['ground_truth'])
            total_wer += wer * len(batch['ground_truth'])
            num_samples += len(batch['ground_truth'])
            
            print(f"Step: {step+1}: CER: {cer:.4f}, WER: {wer:.4f}. Time: {time.time() - batch_time:.2f}s.")
            if step % 10 == 0:
                print('Example of transcription:', translated_gtruth[0])
                print('Predicted transcription:', translated_predictions[0])

    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples

    print(f"Average CER: {avg_cer:.4f}, Average WER: {avg_wer:.4f}")
    print(f'Total time of the evaluation: {time.time() - start_time:.2f}s.')

def main():
    print("Loading data...")
    test_feeder = DataFeeder(mode='test', shuffle_data=False)

    print("Loading model...")
    model_save_path = os.path.join(os.getcwd(), 'model_weights', f'speech_transformer_epoch{LAST_RUN}.pth')
    model = SpeechTransformer(target_vocab_size=test_feeder.vocab_size()).to(DEVICE)
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))

    print("Evaluating model...")
    evaluate_model(model, test_feeder)

if __name__ == "__main__":
    main()