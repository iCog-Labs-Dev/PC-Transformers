import os
import pickle
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from config import Config


class BPE:
    def __init__(self):
        
        os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)
        
        
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train_tokenizer(self):
        """Train the BPE tokenizer on the training subset."""
        train_path = f"{Config.DATA_DIR}/train.txt"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}.")
        
        print("Training tokenizer...")
        with open(train_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        
    
        trainer = trainers.BpeTrainer(
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            vocab_size=Config.VOCAB_SIZE,
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(sentences, trainer=trainer)
        
        
        tokenizer_path = f"{Config.TOKENIZER_DIR}/tokenizer.json"
        self.tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained and saved to {tokenizer_path}.")

    def tokenize_subset(self, subset: str):
        """Tokenize one subset and save the tokenized IDs as .pkl."""
        tokenizer_path = f"{Config.TOKENIZER_DIR}/tokenizer.json"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please train the tokenizer first.")
        
       
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
      
        subset_path = f"{Config.DATA_DIR}/{subset}.txt"
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f"Subset file not found at {subset_path}.")
        
        print(f"Tokenizing {subset} subset...")
        with open(subset_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        
        tokenized = [self.tokenizer.encode(s).ids for s in sentences]
        
        
        tokenized_path = f"{Config.TOKENIZER_DIR}/{subset}_ids.pkl"
        with open(tokenized_path, "wb") as f:
            pickle.dump(tokenized, f)
        print(f"{subset.capitalize()} subset tokenized and saved to {tokenized_path}.")


def process_data():
    """
    Modular function to handle the entire pipeline:
    1. Train the tokenizer (if not already trained).
    2. Tokenize all subsets (train, valid, test).
    """
    processor = BPE()
    tokenizer_path = f"{Config.TOKENIZER_DIR}/tokenizer.json"
    
    if not os.path.exists(tokenizer_path):
        processor.train_tokenizer()
    
    for subset in ["train", "valid", "test"]:
        subset_pkl_path = f"{Config.TOKENIZER_DIR}/{subset}_ids.pkl"
        if not os.path.exists(subset_pkl_path):
            processor.tokenize_subset(subset)



process_data()