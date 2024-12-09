import os
import numpy as np
from tqdm import tqdm
from random import shuffle
from scipy.io import wavfile
from speechpy import processing, feature

# Constants and parameters
DEV_PARTS = ['dev-clean']
TEST_PARTS = ['test-clean']
TRAIN_PARTS = ['train-clean-100', 'train-clean-360']

PAD = 0
SOS = 1
EOS = 2
NUM_MELS = 80
HOP_SIZE = 0.01
BATCH_SIZE = 16
APPLY_CMVN = True
WINDOW_SIZE = 0.025
CHARS = ['<PAD>', '<SOS>', '<EOS>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '\'']

def id2char(ids):
    return ''.join([CHARS[id] for id in ids if id])
class DataFeeder:
    def __init__(self, mode='train', shuffle_data=True, max_seq_len=None): # Passing a value for max_seq_len will make the initiation significantly slower
        self.mode = mode
        self.shuffle = shuffle_data
        self.wav_lst = []
        self.char_list = []
        self.vocab = CHARS
        self.source_init(max_seq_len)

    def source_init(self, max_seq_len):
        print('Initializing data source...')
        if self.mode == 'train':
            parts = TRAIN_PARTS
        elif self.mode == 'dev':
            parts = DEV_PARTS
        elif self.mode == 'test':
            parts = TEST_PARTS
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'dev', 'test'.")

        count = 0 
        lenghts = 0
        for part in parts:
            file_path = os.path.join(os.getcwd(), 'LibriSpeech', 'data', f"{part}.txt")
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file.readlines(), desc=f"Loading {file_path}"):
                    file_id, text = line.strip().split()
                    reader_id, chapter_id, _ = file_id.split('-')
                    
                    wav_file = os.path.join(os.getcwd(), 'LibriSpeech', part, reader_id, chapter_id, f"{file_id}.wav")
                    
                    if max_seq_len:
                        fbank = self.compute_fbank(wav_file)
                        lenghts += len(fbank)
                        count += 1 
                        if len(fbank) <= max_seq_len:
                            self.wav_lst.append(wav_file)
                            self.char_list.append(list(text))
                    else:
                        self.wav_lst.append(wav_file)
                        self.char_list.append(list(text))
                        

        self.data_length = len(self.wav_lst)
        print(f"Loaded {self.data_length} samples.")
        if max_seq_len:
            ...
            print(f'Mean size: {lenghts/count}')

    def char2id(self, line):
        return [self.vocab.index(char) for char in line]

    def wav_padding(self, wav_data_lst):
        max_len = max([len(data) for data in wav_data_lst])
        padded_wavs = np.zeros((len(wav_data_lst), max_len, NUM_MELS, 3), dtype=np.float32)
        
        for i, wav_data in enumerate(wav_data_lst):
            padded_wavs[i, :wav_data.shape[0], :, :] = wav_data

        return padded_wavs
    
    def label_padding(self, label_data_lst):
        max_len = max([len(label) for label in label_data_lst])
        padded_labels = np.zeros((len(label_data_lst), max_len), dtype=np.int32)

        for i, label in enumerate(label_data_lst):
            padded_labels[i, :len(label)] = label

        return padded_labels

    def compute_fbank(self, file):
        sr, signal = wavfile.read(file)
        signal_preemphasized = processing.preemphasis(signal, cof=0.98)
        
        log_fbank = feature.lmfe(signal_preemphasized, 
                                 sampling_frequency=sr, 
                                 frame_length=WINDOW_SIZE,
                                 frame_stride=HOP_SIZE, 
                                 num_filters=NUM_MELS, 
                                 fft_length=512, 
                                 low_frequency=0,
                                 high_frequency=None ) # num_frames x num_filters
        if APPLY_CMVN:
            log_fbank = processing.cmvn(log_fbank, variance_normalization=True)
            
        return feature.extract_derivative_feature(log_fbank)

    def get_batch(self):
        indices = list(range(self.data_length))
        while True:
            if self.shuffle:
                shuffle(indices)

            for i in range(0, self.data_length, BATCH_SIZE):
                batch_indices = indices[i:i + BATCH_SIZE]
                wav_data_lst = []
                label_data_lst = []
                ground_truth_lst = []

                for idx in batch_indices:
                    fbank = self.compute_fbank(self.wav_lst[idx])

                    label = [SOS] + self.char2id(self.char_list[idx]) + [EOS]
                    g_truth = [SOS] + self.char2id(self.char_list[idx]) + [EOS]

                    wav_data_lst.append(fbank)
                    label_data_lst.append(label)
                    ground_truth_lst.append(g_truth)

                pad_label_data = self.label_padding(label_data_lst)
                pad_ground_truth = self.label_padding(ground_truth_lst)
                pad_wav_data = self.wav_padding(wav_data_lst)

                yield {
                    'the_inputs': pad_wav_data,
                    'the_labels': pad_label_data,
                    'ground_truth': pad_ground_truth,
                }

    def __len__(self):
        return self.data_length

if __name__ == '__main__':
    feeder = DataFeeder(mode='dev', shuffle_data=True)
    batch_generator = feeder.get_batch()

    for step, batch in enumerate(batch_generator):
        print(f"Batch {step + 1}")
        print("Inputs shape:", batch['the_inputs'].shape)
        print("Labels shape:", batch['the_labels'].shape)
        print("Ground truth shape:", batch['ground_truth'].shape)
        break
