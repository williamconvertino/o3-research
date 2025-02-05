# main.py
import argparse
import json
import torch
from trainer import GDTransformer, Trainer
from dataset import get_tokenizer, get_dataloaders, prepare_datasets
from generate import evaluate_generation

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True, help="Mode: train or generate")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to model checkpoint")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer.
    tokenizer = get_tokenizer()
    # Instantiate the model.
    model = GDTransformer(config)
    model.to(device)
    
    if args.mode == "train":
        # Get dataloaders for training and validation.
        train_loader, val_loader, test_loader = get_dataloaders(
            tokenizer,
            config["model"]["max_seq_len"],
            config["training"]["batch_size"]
        )
        trainer = Trainer(model, train_loader, val_loader, config, device)
        trainer.train()
    elif args.mode == "generate":
        # Load test dataset for evaluation.
        _, _, test_dataset = prepare_datasets(tokenizer, config["model"]["max_seq_len"])
        # Load the trained model checkpoint.
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device)
        evaluate_generation(model, tokenizer, test_dataset, device=device)
        
if __name__ == "__main__":
    main()
