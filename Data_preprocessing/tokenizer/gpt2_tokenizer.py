import os
import time
import pickle
from transformers import GPT2TokenizerFast
from Data_preprocessing.config import Config
import argparse

"""
This script initializes a GPT-2 tokenizer (Fast version) and prepares it for tokenizing
data from a specified dataset. It adds special tokens, tokenizes train/valid/test
splits, appends EOS tokens, and saves the tokenized sequences as pickle files for later use.

Usage:
    python -m Data_preprocessing.tokenizer.gpt2_tokenizer --dataset=<dataset_name>

Arguments:
    --dataset: Dataset to use (default: opwb)
        Choices:
            ptb  - Penn Treebank dataset
            opwb - Open WebBook dataset
"""

class GPT2TokenizerWrapper:
    def __init__(self, dataset_name=Config.DATASET_NAME):
        print(f"Initializing tokenizer for dataset: {dataset_name}")  
        os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)
        print(f"Ensured tokenizer directory exists: {Config.TOKENIZER_DIR}")  
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.vocab_size = self.tokenizer.vocab_size + len(special_tokens)

        print("Special tokens have been added...") 
        self.dataset_name = dataset_name
        self.tokenizer_path = os.path.join(Config.TOKENIZER_DIR, f"gpt2_tokenizer_{dataset_name}.json")

    def tokenize_and_save(self, subset_name):
        subset_path = os.path.join(Config.DATA_DIR, self.dataset_name, f"{subset_name}.txt")
        if not os.path.exists(subset_path):
           raise FileNotFoundError(f"{subset_name}.txt not found in {os.path.join(Config.DATA_DIR, self.dataset_name)}")

        MAX_SEQ_LENGTH = Config.MAX_LENGTH
        print(f"Reading file: {subset_path}")  # Debug statement

        with open(subset_path, "r", encoding="utf-8") as f:
             sep_id = self.tokenizer.eos_token_id
             tokenized = []
             for line in f:
                 line = line.strip()
                 if not line:
                    continue
                 encoded_line = self.tokenizer.encode(line, truncation=True, max_length=MAX_SEQ_LENGTH - 1)
                 if len(encoded_line) + 1 > MAX_SEQ_LENGTH:
                    encoded_line = encoded_line[:MAX_SEQ_LENGTH - 1]  # Leave space for EOS token
                 encoded_line.append(sep_id)
                 tokenized.append(encoded_line)

    # Validate token IDs
        max_token_id = max(max(seq) for seq in tokenized)
        print(f"Maximum token ID in {subset_name}: {max_token_id}")
        if max_token_id >= self.vocab_size:
           raise ValueError(f"Token ID {max_token_id} exceeds vocabulary size ({self.vocab_size}).")

        output_path = os.path.join(Config.TOKENIZER_DIR, f"{self.dataset_name}_{subset_name}_ids.pkl")
        print(f"Saving tokenized data to: {output_path}")  # Debug statement
        if os.path.exists(output_path):
           print(f"Tokenized IDs already exist for {subset_name} at {output_path}, skipping.")
           return

        with open(output_path, "wb") as f:
            pickle.dump(tokenized, f)
        print(f"Tokenized {subset_name}.txt and saved IDs to {output_path}")
    def save_tokenizer(self):
        print(f"Saving tokenizer configuration to: {self.tokenizer_path}")  
        self.tokenizer.save_pretrained(self.tokenizer_path)
        print(f"Tokenizer saved to {self.tokenizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='opwb', 
        choices=['ptb', 'opwb'], 
        help=(
            "Dataset to use:\n"
            "  ptb  - Penn Treebank dataset\n"
            "  opwb - Open WebBook dataset\n"
        ))
    
    args = parser.parse_args()
    print(f"Using dataset: {args.dataset}")

    print("Starting tokenizer script...")  
    tokenizer_wrapper = GPT2TokenizerWrapper(args.dataset)  
    print("Tokenizer initialized.")  

    for subset_name in ["train", "valid", "test"]:
        print(f"Processing subset: {subset_name}")  
        tokenizer_wrapper.tokenize_and_save(subset_name)

    tokenizer_wrapper.save_tokenizer()
    total_time = time.time() - start_time
    print(f"\nTokenizer completed in {total_time:.2f} seconds")