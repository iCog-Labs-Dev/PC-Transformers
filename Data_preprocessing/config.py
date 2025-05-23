import os

class Config:
    VOCAB_SIZE = 4000
    PAD_ID = 0
    MAX_LENGTH = 64
    
    # Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, "Data", "ptb") 
    TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer", "outputs")  
   
    # Training
    BATCH_SIZE = 8
