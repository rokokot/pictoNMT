# src/pictollms/models/encoders/visual_encoder.py
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        #   frequent compatibility issues are likely caused by the shapes here
        batch_size = x.shape[0]
        x = self.proj(x)     # [batch_size, embed_dim, grid_size, grid_size]
        x = x.flatten(2)     # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)      # [batch_size, n_patches, embed_dim]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout)
            for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)

class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        
        return x

class VisualEncoder(nn.Module):     # init parameters need to be checked thoroughly
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=256, 
                 num_heads=4,
                 mlp_ratio=2.0,
                 dropout=0.1,
                 num_layers=4,
                 output_dim=512):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_layers=num_layers
        )
        
        # Output projection
        self.projection = nn.Linear(embed_dim, output_dim)
        
        # Initialize position embeddings
        self._init_weights()
    
    def _init_weights(self):
        
        position = torch.arange(self.patch_embed.n_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pos_embed.size(-1), 2) * 
                            -(math.log(10000.0) / self.pos_embed.size(-1)))
        
        pos_embed = torch.zeros_like(self.pos_embed[0])
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
    def forward(self, x):       # this fwd pass needs to handle sequece inputs and single image inputs
        original_shape = x.shape
        
        if len(x.shape) == 5:

            # [batch_size, seq_len, channels, height, width]
            
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.view(batch_size * seq_len, *x.shape[2:])
            process_as_sequence = True
        else:
            # [batch_size, channels, height, width]
            batch_size = x.shape[0]
            seq_len = 1
            process_as_sequence = False
        
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        
        output = self.projection(x)
        
        if process_as_sequence:
            output = output.view(batch_size, seq_len, -1)
            
        return output