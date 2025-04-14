# Model/training hyperparameters
class Config:
    # Tokenizer
    VOCAB_SIZE = 10000
    PAD_ID = 0
    MAX_LENGTH = 128
    
    # Data
    DATA_DIR = ".../PC-Transformers/Data_preprocessing/Data/ptb"
    BATCH_SIZE = 32
    TOKENIZER_DIR= "..../PC-Transformers/tokenizer and tokenized data "
    # Training
    DEVICE = "cpu"  