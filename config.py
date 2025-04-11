# Model/training hyperparameters
class Config:
    # Tokenizer
    VOCAB_SIZE = 10000
    PAD_ID = 0
    MAX_LENGTH = 128
    
    # Data
    DATA_DIR = "/home/nardos_tatek/PC-Transformers/Data/ptb"
    BATCH_SIZE = 32
    TOKENIZER_DIR= "/home/nardos_tatek/PC-Transformers/tokenizer/tokenizer"
    # Training
    DEVICE = "cpu"  # Change to "cuda" if GPU available