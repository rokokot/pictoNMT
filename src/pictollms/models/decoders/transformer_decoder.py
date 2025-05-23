# src/pictollms/models/decoders/transformer_decoder.py
import torch
import torch.nn as nn
import math

class TransformerDecoder(nn.Module):
    
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(1000, hidden_size) 
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size,nhead=num_heads,dim_feedforward=hidden_size * 4,dropout=dropout,activation='gelu',batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,num_layers=num_layers)
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target_ids, encoder_outputs, target_mask=None, encoder_mask=None):
        """
        training (teacher forcing)
        
        Args
            target_ids: Target token IDs [batch_size, tgt_len]
            encoder_outputs: Encoder memory bank [batch_size, src_len, hidden_size]
            target_mask: Causal mask for target [tgt_len, tgt_len]
            encoder_mask: Mask for encoder outputs [batch_size, src_len]
        """
        batch_size, tgt_len = target_ids.shape
        
        # Create embeddings
        token_embeds = self.token_embedding(target_ids)
        
        # Add position embeddings
        positions = torch.arange(tgt_len, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create causal mask if not provided
        if target_mask is None:
            target_mask = self._create_causal_mask(tgt_len, target_ids.device)
        
        # Convert encoder mask format
        memory_key_padding_mask = None
        if encoder_mask is not None:
            memory_key_padding_mask = ~encoder_mask.bool()
        
        # Transpose for transformer (expects seq_len, batch, dim)
        embeddings = embeddings.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        # Decoder forward pass
        decoder_output = self.transformer_decoder(
            tgt=embeddings,
            memory=encoder_outputs,
            tgt_mask=target_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Transpose back
        decoder_output = decoder_output.transpose(0, 1)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def _create_causal_mask(self, size, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def generate_step(self, input_ids, encoder_outputs, encoder_mask=None):
        """
        Generate next token logits for beam search
        
        Args
            input_ids: Previous tokens [batch_size, seq_len]
            encoder_outputs: Encoder memory bank [batch_size, src_len, hidden_size]
            encoder_mask: Encoder attention mask [batch_size, src_len]
            
        Outs
            Logits for next token [batch_size, vocab_size]
        """
        with torch.no_grad():
            # Forward pass with current sequence
            logits = self.forward(input_ids, encoder_outputs, encoder_mask=encoder_mask)
            
            # Return logits for last position
            return logits[:, -1, :]  # [batch_size, vocab_size]