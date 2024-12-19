# SpeechTransformer-Pytorch

**Description:**  

This repository contains the source code of an [SpeechTransformer](https://sci-hub.se/downloads/2020-09-03/63/dong2018.pdf) in Pytorch, done for the discipline of Deep Learning from 2024-2 UFV.

---

## Table of Contents
1. [Environment Configuration](#environment-configuration)
2. [Dataset](#dataset)
3. [Scripts](#scripts)

---

## Environment Configuration

### 1. Create the Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
```bash
# For Windows:
venv\Scripts\Activate

# For MacOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

OBS: You also may need to install seperate dependencies, like CUDA for GPU use and ffmpeg for audio conversion.

---

## Dataset

This project used the [LibriSpeech](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf) dataset. 

If you want to run this project, you need to follow these steps to prepare the dataset correctly.
1. Download the datasets at [LibriSpeech](https://www.openslr.org/12) and put them (train-clean-100, dev-other...) inside the LibriSpeech folder.   
2. Open a terminal inside the LibriSpeech folder.  
3. Run the following script to convert the dataset from .flac to .wav.
     ```bash
    python flac2wav.py
    ```
    - **Note:** If you want to preserve the .flac files you need to comment the line 20 of this file.
4. Run the following script to gather all the transcriptions from the different datasets folders. At the end of it, you'll have .txt files inside the folder **data** with pairs of (id transcription), one for each dataset.
     ```bash
    python gather_transcriptions.py
    ```
---

## Scripts
To train and evaluate the Speech-Transformer model, use the following commands:

### Training the Model
Run the train_transformer.py script to train the Speech-Transformer model.
```bash
python train_transformer.py
```

### Evaluating the Model
Run the evaluate_transformer.py script to evaluate the trained model on the test dataset. Adjust the path for the model weights as necessary.
```bash
python evaluate_transformer.py
```

**Note:** For simplification, all the hyperparameters were defined directely in the files. If you want to change them, you'll need to change the files manually.
