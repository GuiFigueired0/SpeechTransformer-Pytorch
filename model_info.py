from torchinfo import summary
from model import SpeechTransformer

def main():
    model = SpeechTransformer(target_vocab_size=31)
    summary(model)

if __name__ == "__main__":
    main()