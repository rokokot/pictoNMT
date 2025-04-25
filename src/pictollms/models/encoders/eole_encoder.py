# src/pictollms/models/encoders/eole_encoder.py
import torch
import torch.nn as nn
from eole.encoders import Encoder
from pictollms.models.encoders.dual_path_encoder import DualPathEncoder

class PictoEoleEncoder(Encoder):
    
    # eole-compatible wrapper for ViT
    
    def __init__(self, opt, embeddings=None):
        super().__init__(opt, embeddings)
        
        # Create visual encoder configuration
        visual_config = {
            'img_size': getattr(opt, 'img_size', 224),
            'patch_size': getattr(opt, 'patch_size', 16),
            'in_channels': getattr(opt, 'in_channels', 3),
            'embed_dim': getattr(opt, 'visual_embed_dim', 192),
            'num_heads': getattr(opt, 'visual_heads', 3),
            'mlp_ratio': getattr(opt, 'visual_mlp_ratio', 2.0),
            'dropout': getattr(opt, 'dropout', [0.1])[0],
            'num_layers': getattr(opt, 'visual_layers', 3),
            'output_dim': getattr(opt, 'visual_dim', 512)
        }
        
        # Create semantic encoder configuration
        semantic_config = {
            'category_vocab_size': getattr(opt, 'category_vocab_size', 200),
            'type_vocab_size': getattr(opt, 'type_vocab_size', 10),
            'embedding_dim': getattr(opt, 'semantic_embed_dim', 256),
            'output_dim': getattr(opt, 'semantic_dim', 512)
        }
        
        # Create fusion configuration
        fusion_config = {
            'input_dim': visual_config['output_dim'] + semantic_config['output_dim'],
            'hidden_dim': getattr(opt, 'fusion_hidden_dim', 512),
            'output_dim': getattr(opt, 'encoder_dim', 512),
            'dropout': getattr(opt, 'dropout', [0.1])[0]
        }
        
        # Create DualPathEncoder
        self.dual_path_encoder = DualPathEncoder(
            visual_config=visual_config,
            semantic_config=semantic_config,
            fusion_config=fusion_config
        )
        
    def forward(self, src, lengths=None):
        """
        Forward pass compatible with Eole's expected format
        
        Args:
            src: Dictionary containing input tensors:
                - src["images"]: Pictogram images
                - src["categories"]: Category indices
                - src["types"]: Type indices
            lengths: Sequence lengths (optional)
            
        Returns:
            Tuple of (encoder_final, memory_bank, lengths) as expected by Eole
        """
        # Extract inputs from src dictionary
        images = src["images"]
        categories = src["categories"]
        types = src["types"]
        
        # Get batch size and sequence length
        batch_size, seq_len = images.shape[0], images.shape[1]
        
        # Create positions based on sequence length
        positions = torch.arange(seq_len, device=images.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get encoder outputs
        encoder_outputs = self.dual_path_encoder(images, categories, types, positions)
        
        # In Eole, we need to return:
        # 1. encoder_final: Final hidden state (we'll use mean pooling)
        # 2. memory_bank: Full sequence of hidden states
        # 3. lengths: Sequence lengths
        
        # Mean pooling for encoder_final
        encoder_final = encoder_outputs.mean(dim=1)
        
        # If lengths not provided, use full sequence length
        if lengths is None:
            lengths = torch.LongTensor([seq_len] * batch_size).to(images.device)
        
        return encoder_final, encoder_outputs, lengths