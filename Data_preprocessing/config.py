import os
# tokenization hyperparameters
class Config:
    # Tokenizer
    VOCAB_SIZE = 10000
    PAD_ID = 0
    MAX_LENGTH = 128
    
    # Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, "Data", "ptb") 
    TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer", "outputs")  
   
    # Training
    BATCH_SIZE = 32
    DEVICE = "cpu"  