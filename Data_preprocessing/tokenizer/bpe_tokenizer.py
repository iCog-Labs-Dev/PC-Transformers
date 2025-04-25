import os
import pickle
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from Data_preprocessing.config import Config

class BPETokenizer:
    def __init__(self):
        """Initialize BPE tokenizer with paths from config"""
        os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train_and_save(self):
        """Train BPE tokenizer and save the model"""
        
        with open(f"{Config.DATA_DIR}/train.txt", "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        
        trainer = trainers.BpeTrainer(
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            vocab_size=Config.VOCAB_SIZE,
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(sentences, trainer=trainer)
        
      
        self.tokenizer.save(f"{Config.TOKENIZER_DIR}/tokenizer.json")

    def tokenize_and_save(self, subset_name):
        """Tokenize a dataset split and save the IDs"""
        
        self.tokenizer = Tokenizer.from_file(f"{Config.TOKENIZER_DIR}/tokenizer.json")
        
       
        with open(f"{Config.DATA_DIR}/{subset_name}.txt", "r", encoding="utf-8") as f:
            tokenized = [self.tokenizer.encode(line.strip()).ids for line in f if line.strip()]
        
       
        with open(f"{Config.TOKENIZER_DIR}/{subset_name}_ids.pkl", "wb") as f:
            pickle.dump(tokenized, f)


bpe = BPETokenizer()
bpe.train_and_save()  
bpe.tokenize_and_save("train")  
bpe.tokenize_and_save("valid")  
bpe.tokenize_and_save("test")   