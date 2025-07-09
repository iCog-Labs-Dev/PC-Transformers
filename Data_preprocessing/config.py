import os

class Config:
    vocab_size = 4000
    max_length = 64
    
    # Data
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(base_dir, "Data", "ptb") 
    tokenizer_dir = os.path.join(base_dir, "tokenizer", "outputs")  
   
    # Training
    batch_size = 8
    num_workers = 8
