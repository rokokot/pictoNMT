# src/pictollms/models/encoders/eole_encoder.py
import torch
import torch.nn as nn
import warnings

# Suppress triton warnings
warnings.filterwarnings('ignore', category=UserWarning, module='triton')

try:
    from eole.eole.encoders.encoder import EncoderBase
    EOLE_BASE_AVAILABLE = True
    print("✅ Using Eole EncoderBase")
except ImportError as e:
    print(f"⚠️ Eole EncoderBase not available: {e}")
    EOLE_BASE_AVAILABLE = False
    EncoderBase = nn.Module

from pictollms.models.encoders.dual_path_encoder import DualPathEncoder

class PictoEoleEncoder(EncoderBase):
    """
    Eole-compatible wrapper for our DualPathEncoder
    """
    
    def __init__(self, opt=None, embeddings=None):
        super().__init__()
        
        # Handle opt parameter (Eole configuration)
        if opt is None:
            # Create default configuration
            class DefaultOpt:
                def __init__(self):
                    self.img_size = 224
                    self.patch_size = 16
                    self.in_channels = 3
                    self.visual_embed_dim = 192
                    self.visual_heads = 3
                    self.visual_mlp_ratio = 2.0
                    self.dropout = [0.1]
                    self.visual_layers = 3
                    self.visual_dim = 512
                    self.category_vocab_size = 200
                    self.type_vocab_size = 10
                    self.semantic_embed_dim = 256
                    self.semantic_dim = 512
                    self.fusion_hidden_dim = 512
                    self.encoder_dim = 512
            
            opt = DefaultOpt()
        
        # Extract configuration with safe defaults
        dropout_val = opt.dropout[0] if isinstance(getattr(opt, 'dropout', [0.1]), list) else getattr(opt, 'dropout', 0.1)
        
        # Create configuration for DualPathEncoder
        visual_config = {
            'img_size': getattr(opt, 'img_size', 224),
            'patch_size': getattr(opt, 'patch_size', 16),
            'in_channels': getattr(opt, 'in_channels', 3),
            'embed_dim': getattr(opt, 'visual_embed_dim', 192),
            'num_heads': getattr(opt, 'visual_heads', 3),
            'mlp_ratio': getattr(opt, 'visual_mlp_ratio', 2.0),
            'dropout': dropout_val,
            'num_layers': getattr(opt, 'visual_layers', 3),
            'output_dim': getattr(opt, 'visual_dim', 512)
        }
        
        semantic_config = {
            'category_vocab_size': getattr(opt, 'category_vocab_size', 200),
            'type_vocab_size': getattr(opt, 'type_vocab_size', 10),
            'embedding_dim': getattr(opt, 'semantic_embed_dim', 256),
            'output_dim': getattr(opt, 'semantic_dim', 512)
        }
        
        fusion_config = {
            'input_dim': visual_config['output_dim'] + semantic_config['output_dim'],
            'hidden_dim': getattr(opt, 'fusion_hidden_dim', 512),
            'output_dim': getattr(opt, 'encoder_dim', 512),
            'dropout': dropout_val
        }
        
        # Create the dual path encoder
        self.dual_path_encoder = DualPathEncoder(
            visual_config=visual_config,
            semantic_config=semantic_config,
            fusion_config=fusion_config
        )
        
        # Store config for reference
        self.config = opt
        
    def forward(self, src, lengths=None):
        """
        Forward pass compatible with Eole's expected format
        
        Args:
            src: Input - can be dict with 'images', 'categories', 'types' keys
                 or tensor (fallback mode)
            lengths: Sequence lengths (optional)
            
        Returns:
            Tuple of (encoder_final, memory_bank, lengths) as expected by Eole
        """
        # Handle input format
        if isinstance(src, dict):
            # Preferred format: dictionary with separate components
            images = src.get("images")
            categories = src.get("categories") 
            types = src.get("types")
            
            if images is None or categories is None or types is None:
                raise ValueError("Dictionary input must contain 'images', 'categories', and 'types' keys")
                
        else:
            # Fallback: assume src is a placeholder, create dummy inputs for testing
            if hasattr(src, 'shape'):
                batch_size = src.shape[0]
                seq_len = src.shape[1] if len(src.shape) > 1 else 1
                device = src.device
            else:
                batch_size, seq_len = 1, 1
                device = torch.device('cpu')
            
            # Create dummy inputs (for testing/fallback only)
            images = torch.randn(batch_size, seq_len, 3, 224, 224, device=device)
            categories = torch.randint(0, 100, (batch_size, seq_len, 5), device=device)
            types = torch.randint(0, 10, (batch_size, seq_len), device=device)
            
            print(f"⚠️ Using fallback mode with dummy inputs for shapes: "
                  f"images={images.shape}, categories={categories.shape}, types={types.shape}")
        
        # Forward through dual path encoder
        encoder_outputs = self.dual_path_encoder(images, categories, types)
        
        # Prepare outputs for Eole compatibility
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # encoder_final: typically mean pooling over sequence
        encoder_final = encoder_outputs.mean(dim=1)  # [batch_size, hidden_dim]
        
        # memory_bank: full sequence outputs
        memory_bank = encoder_outputs  # [batch_size, seq_len, hidden_dim]
        
        # lengths: if not provided, use full sequence length
        if lengths is None:
            lengths = torch.LongTensor([seq_len] * batch_size).to(encoder_outputs.device)
        
        return encoder_final, memory_bank, lengths

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Factory method for Eole compatibility"""
        return cls(opt, embeddings)