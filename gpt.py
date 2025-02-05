# transformer_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    """
    A standard Transformer model following GPT style.
    
    - Input tokens are mapped to embeddings.
    - Learned positional embeddings (of length max_seq_len) are added to the token embeddings.
    - A stack of Transformer blocks is applied.
    - Each block computes queries and keys from the (normalized) combined embeddings,
      but computes values from the raw token embeddings only.
    - A causal mask is used to ensure that each token attends only to previous tokens.
    - A final layer norm and an output projection (via weight tying) produce the logits.
    
    Configuration dictionary (under key "model") is expected to have:
      - vocab_size: int
      - d_model: int (hidden dimension)
      - max_seq_len: int
      - n_layers: int (number of Transformer blocks)
      - n_head: int (number of attention heads)
      - attn_dropout: float (attention dropout rate)
      - ff_dropout: float (dropout rate in the feed-forward network)
      - ff_hidden_mult: int (multiplier for the feed-forward hidden size, typically 4)
    """
    def __init__(self, config):
        super().__init__()
        mconf = config["model"]
        self.vocab_size = mconf["vocab_size"]
        self.d_model = mconf["d_model"]
        self.max_seq_len = mconf["max_seq_len"]
        self.n_layers = mconf["n_layers"]
        self.n_head = mconf["n_head"]
        self.attn_dropout = mconf.get("attn_dropout", 0.1)
        self.ff_dropout = mconf.get("ff_dropout", 0.1)
        self.ff_hidden_mult = mconf.get("ff_hidden_mult", 4)
        
        # Token embeddings (to be tied with output projection)
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        # Positional embeddings (for positions 0 .. max_seq_len-1)
        self.pos_embed = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Transformer blocks (each block is a standard GPT-style block)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_head,
                             self.attn_dropout, self.ff_dropout,
                             self.ff_hidden_mult)
            for _ in range(self.n_layers)
        ])
        # Final layer norm before output projection.
        self.ln_f = nn.LayerNorm(self.d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embeddings similar to GPT-2.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        # Initialize each Transformer block.
        for block in self.blocks:
            block.apply(self._init_block_weights)
        # Initialize final layer norm (scale near 1, bias 0)
        nn.init.constant_(self.ln_f.weight, 1.0)
        nn.init.constant_(self.ln_f.bias, 0.0)
    
    def _init_block_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, targets=None, pad_token_id=None):
        """
        Args:
            x: LongTensor of shape (B, S) containing input token indices.
            targets: (optional) LongTensor of shape (B, S) containing target token indices.
            pad_token_id: (optional) token id to be ignored in loss computation.
            
        Returns:
            If targets is provided:
                logits: Tensor of shape (B, S, vocab_size)
                loss: scalar (cross-entropy loss)
            Otherwise:
                logits: Tensor of shape (B, S, vocab_size)
        """
        B, S = x.size()
        device = x.device
        
        # Token embeddings: shape (B, S, d_model)
        tok_emb = self.token_embed(x)
        # Positional indices: (B, S)
        pos_indices = torch.arange(0, S, device=device).unsqueeze(0).expand(B, -1)
        # Positional embeddings: shape (B, S, d_model)
        pos_emb = self.pos_embed(pos_indices)
        # Combine embeddings (standard GPT-2 uses a sum)
        h = tok_emb + pos_emb  # (B, S, d_model)
        
        # Pass through the Transformer blocks.
        # Note: each block receives the current hidden state h, and also the original
        # token embeddings (tok_emb) to use for the value projection.
        for block in self.blocks:
            h = block(h, tok_emb)
        
        # Final normalization.
        h = self.ln_f(h)
        # Output logits via weight tying.
        logits = torch.matmul(h, self.token_embed.weight.t())
        
        if targets is not None:
            if pad_token_id is None:
                pad_token_id = -1
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.contiguous().view(-1),
                ignore_index=pad_token_id
            )
            return logits, loss
        return logits, None

class TransformerBlock(nn.Module):
    """
    A single Transformer block (GPT style) with:
      - Multi-head self-attention (with causal mask)
      - Residual connections and layer normalization
      - A feed-forward network (MLP) with GELU activation and dropout.
      
    For the self-attention, we compute queries and keys from the normalized hidden state,
    but values are computed from the raw token embeddings passed in as a separate argument.
    """
    def __init__(self, d_model, n_head, attn_dropout, ff_dropout, ff_hidden_mult):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Linear projections for queries and keys (from hidden state)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # Linear projection for values (from token embeddings only)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(attn_dropout)
        
        # Output projection after multi-head attention.
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_dropout = nn.Dropout(attn_dropout)
        
        # Causal mask will be created dynamically based on sequence length.
        # (Alternatively, one could register a buffer with a large mask.)
        
        # Feed-forward network (MLP) with GELU activation.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * d_model, d_model, bias=False),
            nn.Dropout(ff_dropout)
        )
        
        # LayerNorm layers (pre-activation style).
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
    
    def forward(self, x, token_emb):
        """
        Args:
            x: Hidden state, shape (B, S, d_model)
            token_emb: Original token embeddings, shape (B, S, d_model),
                       used to compute the values.
        Returns:
            Updated hidden state, shape (B, S, d_model)
        """
        B, S, _ = x.size()
        
        # === Self-Attention ===
        # Apply layer norm before attention.
        h_norm = self.ln_attn(x)
        
        # Compute queries and keys from the normalized hidden state.
        q = self.q_proj(h_norm)  # (B, S, d_model)
        k = self.k_proj(h_norm)  # (B, S, d_model)
        # Compute values from token embeddings (without positional info).
        v = self.v_proj(token_emb)  # (B, S, d_model)
        
        # Reshape to (B, n_head, S, head_dim)
        q = q.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention.
        attn_scores = torch.matmul(q, k.transpose(-1, -2))  # (B, n_head, S, S)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Create a causal mask: allow each token to attend only to previous tokens (including itself).
        causal_mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply softmax.
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute weighted sum of values.
        attn_output = torch.matmul(attn_weights, v)  # (B, n_head, S, head_dim)
        # Concatenate heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        # Project back to d_model.
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)
        
        # Residual connection.
        x = x + self.resid_dropout(attn_output)
        
        # === Feed-Forward Network ===
        h_ffn = self.ln_ffn(x)
        ffn_output = self.ffn(h_ffn)
        x = x + ffn_output
        
        return x
