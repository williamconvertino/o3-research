# trainer.py
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# GPT-style initialization (no bias)
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

class GDTransformer(nn.Module):
    def __init__(self, config):
        super(GDTransformer, self).__init__()
        model_config = config["model"]
        self.vocab_size = model_config["vocab_size"]
        self.d_model = model_config["d_model"]
        self.num_layers = model_config["num_layers"]
        self.alpha = model_config["alpha"]
        self.max_seq_len = model_config["max_seq_len"]
        self.dropout_rate = model_config.get("dropout", 0.0)
        
        # Token embedding (and tied output projection)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.apply(init_weights)
        
    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        Returns logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Get positional embeddings (same for all samples)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)  # (batch_size, seq_len, d_model)
        pos_emb_single = pos_emb[0]  # (seq_len, d_model)
        
        # Compute a simple linear kernel between positions
        K = torch.matmul(pos_emb_single, pos_emb_single.transpose(0, 1))  # (seq_len, seq_len)
        
        # Get target token embeddings (used in computing the error signal)
        target_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Start with f_0(x)=0
        f = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Each layer is interpreted as one gradient descent update step.
        for _ in range(self.num_layers):
            logits = torch.matmul(f, self.token_embedding.weight.transpose(0, 1))
            probs = F.softmax(logits, dim=-1)
            expected_emb = torch.matmul(probs, self.token_embedding.weight)
            error = target_emb - expected_emb  # (batch_size, seq_len, d_model)
            update = (self.alpha / seq_len) * torch.matmul(K, error)
            update = self.dropout(update)
            f = f + update
        
        final_logits = torch.matmul(f, self.token_embedding.weight.transpose(0, 1))
        return final_logits

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = AdamW(model.parameters(), 
                               lr=config["training"]["learning_rate"],
                               weight_decay=config["training"]["weight_decay"])
        self.num_epochs = config["training"]["num_epochs"]
        
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                total_loss += loss.item()
                total_batches += 1
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def train(self):
        best_val_loss = float("inf")
        total_steps = self.num_epochs * len(self.train_loader)
        current_step = 0
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs+1):
            self.model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                loss.backward()
                # (Optional) gradient clipping can be added here.
                self.optimizer.step()
                
                epoch_loss += loss.item()
                current_step += 1
                
                # Calculate elapsed time and ETA
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / current_step
                remaining_steps = total_steps - current_step
                eta = remaining_steps * avg_time_per_step
                print(f"Epoch {epoch}, Step {current_step}/{total_steps}, Loss: {loss.item():.4f}, ETA: {eta/60:.2f} min", end="\r")
            
            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader)
            print(f"\nEpoch {epoch} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                print("Saved new best model with Val Loss:", best_val_loss)
                
        total_training_time = time.time() - start_time
        print(f"Training complete in {total_training_time/60:.2f} minutes.")
