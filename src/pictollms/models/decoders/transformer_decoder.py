# src/pictollms/models/decoders/transformer_decoder.py
import torch
import torch.nn as nn
import math

class TransformerDecoder(nn.Module):
    """
    Transformer decoder for PictoNMT
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(5000, hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # This is crucial - keep batch first
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, target_ids: torch.Tensor, encoder_outputs: torch.Tensor, 
                target_mask: torch.Tensor = None, encoder_mask: torch.Tensor = None):
        """
        Forward pass for training (teacher forcing)
        
        Args:
            target_ids: Target token IDs [batch_size, tgt_len]
            encoder_outputs: Encoder memory bank [batch_size, src_len, hidden_size]
            target_mask: Causal mask for target [tgt_len, tgt_len] (optional)
            encoder_mask: Mask for encoder outputs [batch_size, src_len]
        
        Returns:
            Logits [batch_size, tgt_len, vocab_size]
        """
        batch_size, tgt_len = target_ids.shape
        src_len = encoder_outputs.shape[1]
        
        # Create token embeddings
        token_embeds = self.token_embedding(target_ids)
        
        # Add position embeddings
        positions = torch.arange(tgt_len, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create causal mask if not provided
        # The mask should be [tgt_len, tgt_len] for the attention mechanism
        if target_mask is None:
            target_mask = self._create_causal_mask(tgt_len, target_ids.device)
        
        # Process encoder mask for memory_key_padding_mask
        # This should be [batch_size, src_len] with True for positions to IGNORE
        memory_key_padding_mask = None
        if encoder_mask is not None:
            # encoder_mask is [batch_size, src_len] with 1 for valid positions
            # We need to invert it for padding mask (True = ignore)
            memory_key_padding_mask = ~encoder_mask.bool()  # [batch_size, src_len]
        
        # Since we're using batch_first=True, we don't need to transpose
        # The transformer expects [batch_size, seq_len, hidden_size]
        
        # Decoder forward pass
        decoder_output = self.transformer_decoder(
            tgt=embeddings,  # [batch_size, tgt_len, hidden_size]
            memory=encoder_outputs,  # [batch_size, src_len, hidden_size]
            tgt_mask=target_mask,  # [tgt_len, tgt_len]
            memory_key_padding_mask=memory_key_padding_mask  # [batch_size, src_len]
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate_step(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor, 
                     encoder_mask: torch.Tensor = None):
        """
        Generate next token logits for beam search
        
        Args:
            input_ids: Previous tokens [batch_size, seq_len]
            encoder_outputs: Encoder memory bank [batch_size, src_len, hidden_size]
            encoder_mask: Encoder attention mask [batch_size, src_len]
            
        Returns:
            Logits for next token [batch_size, vocab_size]
        """
        with torch.no_grad():
            # Forward pass with current sequence
            logits = self.forward(
                target_ids=input_ids,
                encoder_outputs=encoder_outputs,
                target_mask=None,  # Let the forward method create the mask
                encoder_mask=encoder_mask
            )
            
            # Return logits for last position
            return logits[:, -1, :]  # [batch_size, vocab_size]
    
    def _create_causal_mask(self, size: int, device: torch.device):
        """Create causal mask for autoregressive generation"""
        # Create upper triangular matrix with 1s above diagonal
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        # Convert 1s to -inf and 0s to 0 (this prevents attention to future positions)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask