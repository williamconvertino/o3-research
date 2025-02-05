# dataset.py
import os
from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, max_seq_len):
        self.max_seq_len = max_seq_len
        self.examples = []
        # Break the long token sequence into chunks of max_seq_len tokens.
        for i in range(0, len(token_ids) - max_seq_len, max_seq_len):
            self.examples.append(token_ids[i:i+max_seq_len])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.examples[idx], dtype=torch.long)}

def prepare_datasets(tokenizer, max_seq_len, cache_dir="./data"):
    # Load the wikitext-2 raw dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    tokenized_datasets = {}
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        full_text = "\n\n".join(texts)
        tokens = tokenizer.encode(full_text)
        tokenized_datasets[split] = tokens
    
    train_dataset = TextDataset(tokenized_datasets["train"], max_seq_len)
    val_dataset = TextDataset(tokenized_datasets["validation"], max_seq_len)
    test_dataset = TextDataset(tokenized_datasets["test"], max_seq_len)
    
    return train_dataset, val_dataset, test_dataset

def get_tokenizer():
    # Use GPT2Tokenizer; note that GPT2 does not have a PAD token so we add one.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_dataloaders(tokenizer, max_seq_len, batch_size, cache_dir="./data"):
    train_dataset, val_dataset, test_dataset = prepare_datasets(tokenizer, max_seq_len, cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader
