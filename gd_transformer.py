import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ----------------------------
# Configuration loader
# ----------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

# ----------------------------
# Helper: weight initialization (GPT-style)
# ----------------------------
def init_weights(module):
    if isinstance(module, nn.Linear):
        # GPT-2 uses a scaled initialization; here we use Xavier uniform (no bias)
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Embedding):
        # GPT-2 uses a normal distribution with small std for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

# ----------------------------
# Gradient Descent Transformer Model
# ----------------------------
class GDTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config["model"]
        
        self.vocab_size = model_config["vocab_size"]
        self.d_model = model_config["d_model"]
        self.num_layers = model_config["num_layers"]
        self.alpha = model_config["alpha"]
        self.max_seq_len = model_config["max_seq_len"]
        self.num_heads = model_config.get("num_heads", 1)  # for future extension
        self.dropout_rate = model_config.get("dropout", 0.0)
        
        # Token embedding (also used as output projection weight, i.e. weight tying)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Learned positional embeddings.
        # We assume a fixed max sequence length.
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # (Optional) Dropout on the output of each update layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Initialize parameters using a GPT-style scheme
        self.apply(init_weights)
        
        # Note: In this model the learnable parameters are the token and positional embeddings.
        # The “gradient descent” update is performed in the forward pass (without an explicit weight A)
        # and accumulated via skip connections.
        
    def forward(self, input_ids):
        """
        input_ids: LongTensor of shape (batch_size, seq_len)
        Returns: logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Get positional indices (0...seq_len-1) and then the corresponding positional embeddings.
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)  # shape: (batch_size, seq_len, d_model)
        # For our kernel, we assume the positional embeddings are the same for every sample.
        # So we can take the first sample’s positional embeddings:
        pos_emb_single = pos_emb[0]  # shape: (seq_len, d_model)
        
        # Precompute kernel matrix using the positional embeddings.
        # For a linear kernel, k(x_i, x_j) = <x_i, x_j>.
        # We compute a kernel matrix K of shape (seq_len, seq_len)
        # Note: if you wish to extend this with a learned projection (e.g., W_q, W_k), you can do so here.
        K = torch.matmul(pos_emb_single, pos_emb_single.transpose(0, 1))  # (seq_len, seq_len)
        # Optionally, you can scale or normalize K (e.g., similar to scaled dot-product attention)
        
        # Get the ground truth token embeddings for each position (these serve as targets in the GD update)
        # Note: We use the same token embedding table for targets and for the final classification.
        target_emb = self.token_embedding(input_ids)  # shape: (batch_size, seq_len, d_model)
        
        # Initialize the hidden state f to zero (this corresponds to f_0(x) = 0)
        f = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # For each layer, compute the gradient-descent update and add it via a skip connection.
        # We interpret each layer as one “gradient descent” step.
        for layer in range(self.num_layers):
            # Compute logits at current state: shape (batch_size, seq_len, vocab_size)
            # Note: f is (batch, seq_len, d_model) and token_embedding weight is (vocab_size, d_model)
            logits = torch.matmul(f, self.token_embedding.weight.transpose(0, 1))
            
            # Compute predicted distribution for each token (softmax along the vocab dimension)
            # We use dim=-1 since logits shape is (..., vocab_size)
            probs = F.softmax(logits, dim=-1)
            
            # Compute expected token embedding at each position:
            # (batch_size, seq_len, vocab_size) @ (vocab_size, d_model) = (batch_size, seq_len, d_model)
            expected_emb = torch.matmul(probs, self.token_embedding.weight)
            
            # Compute error signal for each position:
            # error = (true token embedding) - (expected token embedding)
            error = target_emb - expected_emb  # shape: (batch_size, seq_len, d_model)
            
            # Now, compute the delta update for each position.
            # According to our formulation, for each output position j:
            # delta[j] = (alpha / seq_len) * sum_{i=1}^{seq_len} error[i] * K[i, j]
            # We can perform this for all positions using a matrix multiply:
            # Let error have shape (batch_size, seq_len, d_model) and K of shape (seq_len, seq_len)
            # Then update = (alpha / seq_len) * (K^T @ error) or simply (K @ error) if K is symmetric.
            update = (self.alpha / seq_len) * torch.matmul(K, error)  # shape: (batch_size, seq_len, d_model)
            update = self.dropout(update)  # apply dropout if desired
            
            # Accumulate the update (skip connection style)
            f = f + update
            
        # Final logits computed from the final f state.
        final_logits = torch.matmul(f, self.token_embedding.weight.transpose(0, 1))
        
        return final_logits