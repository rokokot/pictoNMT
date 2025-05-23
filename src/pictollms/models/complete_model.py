# src/pictollms/models/complete_model.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from pictollms.models.encoders.eole_encoder import PictoEoleEncoder
from pictollms.models.schema.schema_inducer import SchemaInducer
from pictollms.models.decoders.transformer_decoder import TransformerDecoder

class PictoNMT(nn.Module):
    """
    Complete PictoNMT model integrating all components
    """
    
    def __init__(self, vocab_size: int, config=None):
        super().__init__()
        
        if config is None:
            config = self._default_config()
        
        self.config = config
        
        # Extract dropout value (handle both Eole format [0.1] and standard format 0.1)
        dropout_val = config.dropout[0] if isinstance(config.dropout, list) else config.dropout
        
        # Initialize encoder
        self.encoder = PictoEoleEncoder(config)
        
        # Initialize schema inducer
        self.schema_inducer = SchemaInducer(
            hidden_size=config.encoder_dim,
            num_heads=config.schema_heads,
            num_layers=config.schema_layers
        )
        
        # Initialize decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=config.encoder_dim,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=dropout_val  # Use extracted float value
        )
        
        # Store vocab size
        self.vocab_size = vocab_size
    
    def _default_config(self):
        """Create default configuration"""
        class Config:
            def __init__(self):
                # Encoder configuration
                self.encoder_dim = 512
                self.img_size = 224
                self.patch_size = 16
                self.in_channels = 3
                self.visual_embed_dim = 192
                self.visual_heads = 3
                self.visual_mlp_ratio = 2.0
                self.visual_layers = 3
                self.visual_dim = 512
                self.category_vocab_size = 200
                self.type_vocab_size = 10
                self.semantic_embed_dim = 256
                self.semantic_dim = 512
                self.fusion_hidden_dim = 512
                
                # Schema inducer configuration
                self.schema_heads = 8
                self.schema_layers = 3
                
                # Decoder configuration
                self.decoder_layers = 6
                self.decoder_heads = 8
                
                # General configuration (use float, not list)
                self.dropout = 0.1
        
        return Config()
    
    def forward(self, batch: Dict[str, torch.Tensor], mode: str = 'train'):
      """
      Forward pass through the complete model
      """
      # Prepare encoder inputs
      encoder_inputs = {
          'images': batch['images'],
          'categories': batch['categories'],
          'types': batch['types']
      }
      
      # Encode pictogram sequence
      encoder_final, memory_bank, lengths = self.encoder(encoder_inputs)
      
      # Generate schema
      schema = self.schema_inducer(memory_bank, batch.get('attention_masks'))
      
      if mode == 'train':
          # Training mode - use teacher forcing
          target_ids = batch['target_ids']
          
          # Decoder forward pass (remove last token from input for teacher forcing)
          logits = self.decoder(
              target_ids=target_ids[:, :-1],  # Remove last token
              encoder_outputs=memory_bank,
              target_mask=None,  # Let decoder create its own mask
              encoder_mask=batch.get('attention_masks')
          )
          
          return {
              'logits': logits,
              'schema': schema,
              'memory_bank': memory_bank,
              'encoder_final': encoder_final
          }
      
      elif mode == 'inference':
          # Inference mode - return components for beam search
          return {
              'encoder_outputs': memory_bank,
              'schema': schema,
              'encoder_mask': batch.get('attention_masks'),
              'encoder_final': encoder_final
          }
      
      else:
          raise ValueError(f"Unknown mode: {mode}")
    
    def generate(self, batch: Dict[str, torch.Tensor], beam_search, tokenizer, max_length: int = 100):
        """
        Generate translations using beam search
        
        Args:
            batch: Input batch
            beam_search: CASI beam search instance
            tokenizer: Tokenizer for decoding
            max_length: Maximum generation length
        
        Returns:
            List of generated sequences
        """
        self.eval()
        
        with torch.no_grad():
            # Get model outputs for inference
            outputs = self.forward(batch, mode='inference')
            
            # Generate using beam search
            sequences = beam_search.search(
                model=self,
                encoder_outputs=outputs['encoder_outputs'],
                schema=outputs['schema'],
                tokenizer=tokenizer,
                attention_mask=outputs['encoder_mask']
            )
            
            return sequences