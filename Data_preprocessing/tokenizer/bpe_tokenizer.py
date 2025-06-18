import os
import pickle
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from ..config import Config

"""Usage: python -m Data_preprocessing.tokenizer.bpe_tokenizer"""
class BPETokenizer:
    def __init__(self):
        """Initialize BPE tokenizer with paths from config"""
        os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]

    def train_and_save(self):
        """Train BPE tokenizer and save the model"""
        tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
        
        if os.path.exists(tokenizer_path):
            print(f"Tokenizer already exists at {tokenizer_path}, skipping training.")
            return 
        
        train_path = os.path.join(Config.DATA_DIR, "train.txt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        
        with open(train_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        trainer = trainers.BpeTrainer(
            special_tokens=self.special_tokens,
            vocab_size=Config.VOCAB_SIZE,
            min_frequency=2
        )
        self.tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)

        metadata = {
            "special_tokens": self.special_tokens
        }
        metadata_path = os.path.join(Config.TOKENIZER_DIR, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Tokenizer trained and saved to {tokenizer_path}")
        print(f"Metadata saved to {metadata_path}")

    def tokenize_and_save(self, subset_name):
        """Tokenize a dataset split and save the IDs"""
        tokenizer_path = os.path.join(Config.TOKENIZER_DIR, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        subset_path = os.path.join(Config.DATA_DIR, f"{subset_name}.txt")
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f"{subset_name}.txt not found in {Config.DATA_DIR}")
        
        with open(subset_path, "r", encoding="utf-8") as f:
            sep_id = self.tokenizer.token_to_id("[EOS]") or self.tokenizer.token_to_id("[SEP]")
            if sep_id is None:
                raise ValueError("Special token [EOS] or [SEP] not found in tokenizer vocabulary.")

            tokenized = [
                self.tokenizer.encode(line.strip()).ids + [sep_id]
                for line in f if line.strip()
            ]

        output_path = os.path.join(Config.TOKENIZER_DIR, f"{subset_name}_ids.pkl")
        if os.path.exists(output_path):
            print(f"Tokenized IDs already exist for {subset_name} at {output_path}, skipping.")
            return

        with open(output_path, "wb") as f:
            pickle.dump(tokenized, f)

        print(f"Tokenized {subset_name}.txt and saved IDs to {output_path}")
      
bpe = BPETokenizer()
bpe.train_and_save()  
bpe.tokenize_and_save("train")  
bpe.tokenize_and_save("valid")  
bpe.tokenize_and_save("test")   