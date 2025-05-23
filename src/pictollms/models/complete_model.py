import torch
import torch.nn as nn

from pictollms.models.encoders.eole_encoder import PictoEoleEncoder
from pictollms.models.schema.schema_inducer import SchemaInducer
from pictollms.models.decoders.transformer_decoder import TransformerDecoder

class PictoNMT(nn.Module):
    
    def __init__(self, vocab_size, config=None):
        super().__init__()
        
        if config is None:
            config = self._default_config()
        
        self.encoder = PictoEoleEncoder(config)
        self.schema_inducer = SchemaInducer(hidden_size=config.encoder_dim,num_heads=config.schema_heads,num_layers=config.schema_layers)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=config.encoder_dim,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.dropout
        )
        
        self.config = config
    
    def _default_config(self):
        class Config:
            def __init__(self):
                self.encoder_dim = 512
                self.schema_heads = 8
                self.schema_layers = 3
                self.decoder_layers = 6
                self.decoder_heads = 8
                self.dropout = 0.1
        return Config()
    
    def forward(self, batch, mode='train'):
        """
        Forward pass
        
        Args:
            batch: Input batch with images, categories, types, targets
            mode: 'train' or 'inference'
        """
        # Prepare encoder inputs
        encoder_inputs = {'images': batch['images'],
            'categories': batch['categories'],
            'types': batch['types']}
        
        encoder_final, memory_bank, lengths = self.encoder(encoder_inputs)
        
        schema = self.schema_inducer(memory_bank, batch['attention_masks'])
        
        if mode == 'train':
            # teacher forcing training
            logits = self.decoder(
                target_ids=batch['target_ids'][:, :-1],  # Remove last token
                encoder_outputs=memory_bank,
                encoder_mask=batch['attention_masks']
            )
            return {from pictollms.models.encoders.eole_encoder import PictoEoleEncoder
                'logits': logits,
                'schema': schema,
                'memory_bank': memory_bank
            }
        
        else:
            # pass to inference
            return {
                'encoder_outputs': memory_bank,
                'schema': schema,
                'encoder_mask': batch['attention_masks']
            }