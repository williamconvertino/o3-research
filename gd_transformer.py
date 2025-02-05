# gd_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GDTransformer(nn.Module):
    """
    Gradient Descent-based Language Model.
    
    This model interprets the forward pass as performing iterative gradient descent
    updates on a latent function. It uses learned token embeddings and positional
    embeddings. At each layer (gradient descent step), it computes an error signal
    as the difference between the true token embeddings and an expectation computed
    from the current latent state. This error is then aggregated via an attention
    mechanism (with either softmax or linear attention) computed from full query/key
    projections of the positional embeddings.
    
    The positional embeddings are shifted so that the keys come from positions
    0...S-1 and the queries from positions 1...S+1, meaning that the final (N+1th)
    positional output is used for prediction.
    """
    
    def __init__(self, config):
        """
        Args:
            config (dict): A configuration dictionary with keys in "model".
              Expected keys include:
                - vocab_size: vocabulary size (int)
                - d_model: embedding dimension (int)
                - max_seq_len: maximum input sequence length (int)
                - n_layers: number of gradient descent update steps (int)
                - attn_fn: attention function to use ("softmax" or "linear")
                - attn_dropout: dropout rate for attention weights (float)
                - gd_dropout: dropout rate for the gradient descent update (float)
        """
        super().__init__()
        mconf = config["model"]
        self.vocab_size = mconf["vocab_size"]
        self.d_model = mconf["d_model"]
        self.max_seq_len = mconf["max_seq_len"]
        self.n_layers = mconf["n_layers"]
        self.attn_fn = mconf.get("attn_fn", "softmax")  # "softmax" or "linear"
        self.attn_dropout_rate = mconf.get("attn_dropout", 0.1)
        self.gd_dropout_rate = mconf.get("gd_dropout", 0.1)
        
        # Token embeddings (to be used for both encoding and output projection)
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional embeddings: we allocate max_seq_len+1 positions so that we can use
        # the (S+1)th position for prediction.
        self.pos_embed = nn.Embedding(self.max_seq_len + 1, self.d_model)
        
        # Query and Key projections – using full learnable matrices.
        # They project a positional embedding (of dimension d_model) into a d_model space.
        self.linear_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.linear_k = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # For each gradient descent update (i.e. layer), we have an output projection.
        # This projects the aggregated update (of shape d_model) back to d_model.
        self.out_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model, bias=False)
            for _ in range(self.n_layers)
        ])
        
        # Dropout modules for attention and GD update.
        self.attn_dropout = nn.Dropout(self.attn_dropout_rate)
        self.gd_dropout = nn.Dropout(self.gd_dropout_rate)
        
        # Final layer normalization applied to the latent state before computing logits.
        self.ln_out = nn.LayerNorm(self.d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize token and positional embeddings.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        
        # Initialize the linear layers (query, key, and output projections).
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        for proj in self.out_projs:
            nn.init.xavier_uniform_(proj.weight)
    
    def compute_expectation(self, f_state):
        """
        Computes the expectation E[W] given the latent state f_state.
        
        Args:
            f_state: Tensor of shape (B, L, d_model) corresponding to the latent state
                     for token positions.
        
        Returns:
            expectation: Tensor of shape (B, L, d_model)
            
        The calculation is:
            logits = f_state @ (token_embed)^T
            g = exp(logits - max(logits))    [for numerical stability]
            expectation = (g @ token_embed) / sum(g)
        """
        # f_state: (B, L, d_model)
        logits = torch.matmul(f_state, self.token_embed.weight.t())  # (B, L, vocab_size)
        # Numerical stability: subtract the maximum along the vocab dimension.
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        g = torch.exp(logits_stable)  # (B, L, vocab_size)
        numerator = torch.matmul(g, self.token_embed.weight)  # (B, L, d_model)
        denominator = g.sum(dim=-1, keepdim=True) + 1e-8
        expectation = numerator / denominator
        return expectation
    
    def compute_attention(self, queries, keys):
        """
        Computes the attention weights from queries and keys.
        
        Args:
            queries: Tensor of shape (B, L, d_model)
            keys:    Tensor of shape (B, L, d_model)
            
        Returns:
            attn_weights: Tensor of shape (B, L, L)
            
        The attention scores are computed as:
            scores = (queries @ keys^T) / sqrt(d_model)
        and then a causal mask is applied so that for each output position i,
        only keys from positions <= i are attended to. For "softmax" attention,
        softmax is applied; for "linear" attention, the masked positions are set to 0.
        """
        B, L, _ = queries.size()
        scaling = 1.0 / math.sqrt(self.d_model)
        # Dot-product attention scores: (B, L, L)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) * scaling
        
        # Build a causal mask: allow each query to attend only to keys at positions <= its index.
        # mask[i, j] = True if j > i.
        causal_mask = torch.triu(torch.ones(L, L, device=queries.device, dtype=torch.bool), diagonal=1)
        
        if self.attn_fn == "softmax":
            # For softmax attention, set disallowed scores to -inf.
            scores = scores.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
        elif self.attn_fn == "linear":
            # For linear attention, simply zero out the scores for disallowed positions.
            scores = scores.masked_fill(causal_mask, 0.0)
            attn_weights = scores
        else:
            raise ValueError(f"Unknown attention function: {self.attn_fn}")
        
        attn_weights = self.attn_dropout(attn_weights)
        return attn_weights
    
    def forward(self, x, targets=None, pad_token_id=None):
        """
        Args:
            x: LongTensor of shape (B, S) containing token indices.
            targets: (optional) LongTensor of shape (B, S) containing target indices.
            pad_token_id: (optional) integer to be ignored in loss.
            
        Returns:
            If targets is provided:
                logits: Tensor of shape (B, S, vocab_size)
                loss: scalar loss value.
            Otherwise:
                logits: Tensor of shape (B, vocab_size) (from the final output position)
        """
        B, S = x.shape
        device = x.device
        
        # Get token embeddings: (B, S, d_model)
        e = self.token_embed(x)
        
        # Get positional embeddings for positions 0 ... S (total S+1 positions).
        pos_indices = torch.arange(0, S+1, device=device)  # shape: (S+1,)
        # p: (B, S+1, d_model)
        p = self.pos_embed(pos_indices).unsqueeze(0).expand(B, -1, -1)
        
        # Initialize latent state f with zeros: shape (B, S+1, d_model)
        # f[0] will serve as f_0; updates will be added to positions 1 ... S+1.
        f = torch.zeros(B, S+1, self.d_model, device=device)
        
        # Perform n_layers of gradient descent updates.
        for layer in range(self.n_layers):
            # Compute expectation E[W] from the latent state for token positions:
            # Use f[:, :S, :] corresponding to positions 0 ... S-1.
            f_tokens = f[:, :S, :]  # (B, S, d_model)
            E_w = self.compute_expectation(f_tokens)  # (B, S, d_model)
            
            # Error signal: difference between the true token embedding (e) and expectation.
            # (B, S, d_model)
            error = e - E_w
            
            # Compute keys and queries from the positional embeddings.
            # Keys come from positions 0 ... S-1; queries come from positions 1 ... S.
            keys = self.linear_k(p[:, :S, :])    # (B, S, d_model)
            queries = self.linear_q(p[:, 1:S+1, :])  # (B, S, d_model)
            
            # Compute attention weights: (B, S, S)
            attn_weights = self.compute_attention(queries, keys)
            
            # Compute the update as a weighted sum of the error.
            # For each query position j (positions 1...S), the update is:
            #   update_j = sum_{i=0}^{S-1} attn_weights[j, i] * error[i]
            update = torch.matmul(attn_weights, error)  # (B, S, d_model)
            update = self.gd_dropout(update)
            
            # Add the update to positions 1 ... S+1 in the latent state.
            # f[:, 1:S+1, :] = f[:, 1:S+1, :] + update
            f[:, 1:S+1, :] = f[:, 1:S+1, :] + update
        
        # Apply final layer normalization.
        f_norm = self.ln_out(f)
        
        if targets is not None:
            # When training, use positions 1 ... S as predictions for tokens 0 ... S-1.
            logits = torch.matmul(f_norm[:, 1:S+1, :], self.token_embed.weight.t())  # (B, S, vocab_size)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.contiguous().view(-1),
                                   ignore_index=pad_token_id)
            return logits, loss
        else:
            # For generation, use the final position (S+1) to produce logits.
            logits = torch.matmul(f_norm[:, -1, :], self.token_embed.weight.t())  # (B, vocab_size)
            return logits, None

# --- Sample configuration JSON (config.json) ---
# You can use a JSON file similar to the one below.
#
# {
#   "model": {
#     "vocab_size": 50257,
#     "d_model": 512,
#     "max_seq_len": 256,
#     "n_layers": 4,
#     "attn_fn": "softmax",
#     "attn_dropout": 0.1,
#     "gd_dropout": 0.1
#   },
#   "training": {
#     "batch_size": 16,
#     "learning_rate": 0.001,
#     "weight_decay": 0.01,
#     "num_epochs": 10
#   }
# }
