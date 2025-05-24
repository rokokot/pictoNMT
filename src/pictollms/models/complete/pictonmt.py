# src/pictollms/models/complete/pictonmt.py
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
    Fixed version with proper decoder integration
    """
    
    def __init__(self, vocab_size: int, config: Dict):
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        self.pad_token_id = 0  # Will be updated by tokenizer
        
        # Dual-Path Encoder Components
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Position encoding for encoder outputs
        self.encoder_position_encoding = PositionalEncoding(
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
        
        # Transformer Decoder - FIXED INTEGRATION
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=config.get('decoder_layers', 6),
            num_heads=config.get('decoder_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        # Initialize search strategies
        self._initialize_search_strategies()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _initialize_search_strategies(self):
        """Initialize search strategy objects"""
        self.search_strategies = {
            'greedy': GreedySearch(max_length=100),
            'beam': BeamSearch(beam_size=4, max_length=100),
            'casi': CAsiBeamSearch(beam_size=4, max_length=100, schema_weight=0.3)
        }
    
    def forward(self, batch: Dict[str, torch.Tensor], mode: str = 'train'):
        """Forward pass through the complete model"""
        
        # Extract components from batch
        images = batch['images']
        categories = batch.get('categories', torch.zeros(images.shape[0], images.shape[1], 5, dtype=torch.long, device=images.device))
        types = batch.get('types', torch.zeros(images.shape[0], images.shape[1], dtype=torch.long, device=images.device))
        attention_mask = batch.get('attention_mask')
        
        # Dual-path encoding
        visual_features = self.visual_encoder(images)  # [batch_size, seq_len, visual_dim]
        semantic_features = self.semantic_encoder(categories, types)  # [batch_size, seq_len, semantic_dim]
        
        # Fusion
        combined_features = torch.cat([visual_features, semantic_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        encoder_outputs = self.encoder_position_encoding(fused_features)
        
        # Schema induction
        schema = self.schema_inducer(encoder_outputs, attention_mask)
        enhanced_outputs = schema['enhanced_repr']
        
        if mode == 'train':
            # Training with teacher forcing
            target_ids = batch['target_ids']
            
            # FIXED: Proper decoder forward pass
            logits = self.decoder(
                target_ids=target_ids[:, :-1],  # Remove last token for input
                encoder_outputs=enhanced_outputs,
                target_mask=None,  # Let decoder create causal mask
                encoder_mask=attention_mask
            )
            
            return {
                'logits': logits,
                'schema': schema,
                'encoder_outputs': enhanced_outputs,
                'attention_mask': attention_mask
            }
        else:
            # Return components for generation
            return {
                'encoder_outputs': enhanced_outputs,
                'schema': schema,
                'attention_mask': attention_mask
            }
    
    def generate(self, batch: Dict[str, torch.Tensor], strategy: str = 'greedy', tokenizer=None):
        """Generate translations using specified strategy"""
        
        if tokenizer is None:
            raise ValueError("Tokenizer is required for generation")
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, mode='inference')
            
            # Update pad token ID
            self.pad_token_id = tokenizer.pad_token_id
            
            if strategy not in self.search_strategies:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.search_strategies.keys())}")
            
            search = self.search_strategies[strategy]
            
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
        """Compute training loss with proper handling"""
        logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
        target_ids = batch['target_ids']  # [batch_size, seq_len]
        
        # Align target with logits (shift by 1 for next token prediction)
        target_flat = target_ids[:, 1:].contiguous().view(-1)  # Remove first token, flatten
        logits_flat = logits.contiguous().view(-1, logits.size(-1))  # Flatten
        
        # Compute cross-entropy loss
        main_loss = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.pad_token_id
        )
        
        # FIXED: Add schema auxiliary losses
        schema = outputs['schema']
        aux_losses = self._compute_schema_losses(schema, batch)
        
        # Combine losses
        total_loss = main_loss + 0.1 * aux_losses['total']
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'schema_loss': aux_losses['total'],
            'structure_loss': aux_losses.get('structure', 0),
            'functional_loss': aux_losses.get('functional', 0)
        }
    
    def _compute_schema_losses(self, schema: Dict, batch: Dict):
        """Compute auxiliary losses from schema predictions"""
        losses = {}
        total_aux_loss = 0
        
        # Structure type prediction loss (simplified)
        if 'structure_logits' in schema:
            # Create pseudo ground truth based on sequence length (simplified)
            batch_size = schema['structure_logits'].shape[0]
            seq_lengths = batch.get('attention_mask', torch.ones(batch_size, 10)).sum(dim=1)
            
            # Simple heuristic: longer sequences -> more complex structures
            pseudo_targets = torch.clamp((seq_lengths - 3), 0, 9).long().to(schema['structure_logits'].device)
            
            structure_loss = F.cross_entropy(schema['structure_logits'], pseudo_targets)
            losses['structure'] = structure_loss
            total_aux_loss += structure_loss
        
        # Functional word prediction losses
        functional_loss = 0
        if 'determiner_logits' in schema:
            # Simple regularization loss to encourage diversity
            det_logits = schema['determiner_logits']
            functional_loss += -torch.mean(torch.sum(F.log_softmax(det_logits, dim=-1) * F.softmax(det_logits, dim=-1), dim=-1))
        
        if functional_loss > 0:
            losses['functional'] = functional_loss
            total_aux_loss += 0.5 * functional_loss
        
        losses['total'] = total_aux_loss
        return losses
    
    def get_model_size(self):
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_breakdown = {
            'visual_encoder': sum(p.numel() for p in self.visual_encoder.parameters()),
            'semantic_encoder': sum(p.numel() for p in self.semantic_encoder.parameters()),
            'fusion_layer': sum(p.numel() for p in self.fusion_layer.parameters()),
            'schema_inducer': sum(p.numel() for p in self.schema_inducer.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters())
        }
        
        return f"{total_params:,} total ({trainable_params:,} trainable) - " + \
               f"Visual: {param_breakdown['visual_encoder']:,}, " + \
               f"Semantic: {param_breakdown['semantic_encoder']:,}, " + \
               f"Schema: {param_breakdown['schema_inducer']:,}, " + \
               f"Decoder: {param_breakdown['decoder']:,}"


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
        # Image processing
        'img_size': 224,
        'patch_size': 16,
        
        # Visual encoder
        'visual_embed_dim': 256,
        'visual_heads': 8,
        'visual_mlp_ratio': 4.0,
        'visual_layers': 6,
        'visual_output_dim': 512,
        
        # Semantic encoder
        'category_vocab_size': 200,
        'type_vocab_size': 10,
        'semantic_embed_dim': 256,
        'semantic_output_dim': 512,
        
        # Model architecture
        'hidden_dim': 512,
        'max_seq_len': 100,
        'dropout': 0.1,
        
        # Schema inducer
        'schema_heads': 8,
        'schema_layers': 3,
        
        # Decoder
        'decoder_layers': 6,
        'decoder_heads': 8,
        
        # Generation
        'beam_size': 4,
        'max_length': 100,
        'length_penalty': 0.6,
        'schema_weight': 0.3
    }