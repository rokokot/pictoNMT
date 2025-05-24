import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from pictollms.models.encoders.visual_encoder import VisualEncoder
from pictollms.models.encoders.semantic_encoder import SemanticEncoder
from pictollms.models.schema.schema_inducer import SchemaInducer
from pictollms.models.decoders.transformer_decoder import TransformerDecoder
from pictollms.decoding.beam_search import GreedySearch, BeamSearch, CAsiBeamSearch
        

class PictoNMT(nn.Module):
    """
    Complete PictoNMT system with dual-path encoding and schema induction
    """
    
    def __init__(self, vocab_size: int, config: Dict):
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        
        # Dual-Path Encoder
        self.visual_encoder = VisualEncoder(
            img_size=config.get('img_size', 224),
            patch_size=config.get('patch_size', 16),
            in_channels=3,
            embed_dim=config.get('visual_embed_dim', 256),
            num_heads=config.get('visual_heads', 8),
            mlp_ratio=config.get('visual_mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            num_layers=config.get('visual_layers', 6),
            output_dim=config.get('visual_output_dim', 512)
        )
        
        self.semantic_encoder = SemanticEncoder(
            category_vocab_size=config.get('category_vocab_size', 200),
            type_vocab_size=config.get('type_vocab_size', 10),
            embedding_dim=config.get('semantic_embed_dim', 256),
            output_dim=config.get('semantic_output_dim', 512)
        )
        
        # Fusion layer
        hidden_dim = config.get('hidden_dim', 512)
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, hidden_dim),  # visual_dim + semantic_dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position encoding
        self.position_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=config.get('dropout', 0.1),
            max_len=config.get('max_seq_len', 100)
        )
        
        # Schema Induction Module
        self.schema_inducer = SchemaInducer(
            hidden_size=hidden_dim,
            num_heads=config.get('schema_heads', 8),
            num_layers=config.get('schema_layers', 3)
        )
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=config.get('decoder_layers', 6),
            num_heads=config.get('decoder_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        # Store pad token for loss computation
        self.pad_token_id = 0
    
    def forward(self, batch: Dict[str, torch.Tensor], mode: str = 'train'):
        """Forward pass through the complete model"""
        
        # Dual-path encoding
        visual_features = self.visual_encoder(batch['images'])
        semantic_features = self.semantic_encoder(batch['categories'], batch['types'])
        
        # Fusion
        combined_features = torch.cat([visual_features, semantic_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        encoder_outputs = self.position_encoding(fused_features)
        
        # Schema induction
        schema = self.schema_inducer(encoder_outputs, batch.get('attention_mask'))
        enhanced_outputs = schema['enhanced_repr']
        
        if mode == 'train':
            # Training with teacher forcing
            target_ids = batch['target_ids']
            logits = self.decoder(
                target_ids=target_ids[:, :-1],
                encoder_outputs=enhanced_outputs,
                encoder_mask=batch.get('attention_mask')
            )
            
            return {
                'logits': logits,
                'schema': schema,
                'encoder_outputs': enhanced_outputs
            }
        else:
            # Return components for generation
            return {
                'encoder_outputs': enhanced_outputs,
                'schema': schema,
                'attention_mask': batch.get('attention_mask')
            }
    
    def generate(self, batch: Dict[str, torch.Tensor], strategy: str = 'greedy', tokenizer=None):
        """Generate translations"""
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, mode='inference')
            
            if strategy == 'greedy':
                search = GreedySearch(max_length=100)
            elif strategy == 'beam':
                search = BeamSearch(beam_size=4, max_length=100)
            elif strategy == 'casi':
                search = CAsiBeamSearch(beam_size=4, max_length=100, schema_weight=0.3)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if strategy == 'casi':
                return search.search(
                    decoder=self.decoder,
                    encoder_outputs=outputs['encoder_outputs'],
                    schema=outputs['schema'],
                    attention_mask=outputs['attention_mask'],
                    tokenizer=tokenizer
                )
            else:
                return search.search(
                    decoder=self.decoder,
                    encoder_outputs=outputs['encoder_outputs'],
                    attention_mask=outputs['attention_mask'],
                    tokenizer=tokenizer
                )
    
    def compute_loss(self, outputs: Dict, batch: Dict):
        """Compute training loss"""
        logits = outputs['logits']
        target_ids = batch['target_ids']
        
        # Cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        target_flat = target_ids[:, 1:].contiguous().view(-1)
        logits_flat = logits.view(-1, vocab_size)
        
        loss = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.pad_token_id
        )
        
        return {'total_loss': loss, 'main_loss': loss}


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def create_model_config(vocab_size: int) -> Dict:
    """Create default model configuration"""
    return {
        'img_size': 224,
        'patch_size': 16,
        'visual_embed_dim': 256,
        'visual_heads': 8,
        'visual_mlp_ratio': 4.0,
        'visual_layers': 6,
        'visual_output_dim': 512,
        'category_vocab_size': 200,
        'type_vocab_size': 10,
        'semantic_embed_dim': 256,
        'semantic_output_dim': 512,
        'hidden_dim': 512,
        'max_seq_len': 100,
        'schema_heads': 8,
        'schema_layers': 3,
        'decoder_layers': 6,
        'decoder_heads': 8,
        'dropout': 0.1,
        'beam_size': 4,
        'max_length': 100,
        'length_penalty': 0.6,
        'schema_weight': 0.3
    }