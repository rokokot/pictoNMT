# src/pictollms/models/encoders/dual_path_encoder.py
import torch
import torch.nn as nn
from pictollms.models.encoders.visual_encoder import VisualEncoder
from pictollms.models.encoders.semantic_encoder import SemanticEncoder

class DualPathEncoder(nn.Module):
    
    def __init__(self, 
                visual_config=None,
                semantic_config=None,
                fusion_config=None):
        super().__init__()
        
        visual_config = visual_config or {
            'img_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'embed_dim': 192,
            'num_heads': 3,
            'mlp_ratio': 2.0,
            'dropout': 0.1,
            'num_layers': 3,
            'output_dim': 512
        }
        
        semantic_config = semantic_config or {
            'category_vocab_size': 200,
            'type_vocab_size': 10,
            'embedding_dim': 256,
            'output_dim': 512
        }
        
        fusion_config = fusion_config or {
            'input_dim': visual_config['output_dim'] + semantic_config['output_dim'],
            'hidden_dim': 512,
            'output_dim': 512,
            'dropout': 0.1
        }
        
        self.visual_encoder = VisualEncoder(**visual_config)
        
        self.semantic_encoder = SemanticEncoder(**semantic_config)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_config['input_dim'], fusion_config['hidden_dim']),
            nn.LayerNorm(fusion_config['hidden_dim']),
            nn.Dropout(fusion_config['dropout']),
            nn.ReLU(),
            nn.Linear(fusion_config['hidden_dim'], fusion_config['output_dim']),
            nn.LayerNorm(fusion_config['output_dim']),
            nn.Dropout(fusion_config['dropout'])
        )
        
        self.position_encoding = nn.Embedding(1000, fusion_config['output_dim'])  # Max 1000 positions
        
    def forward(self, images, categories, types, positions=None):
        visual_features = self.visual_encoder(images)
        
        semantic_features = self.semantic_encoder(categories, types)
        
        combined = torch.cat([visual_features, semantic_features], dim=2)
        
        fused = self.fusion(combined)
        
        if positions is None:
            batch_size, seq_len = images.shape[0], images.shape[1]
            positions = torch.arange(seq_len, device=images.device).unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = self.position_encoding(positions)
        
        output = fused + position_embeddings
        
        return output