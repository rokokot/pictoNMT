# tests/test_picto_encoder.py
import torch
import torch.nn as nn
from pictollms.models.encoders.visual_encoder import VisualEncoder

def test_vit_encoder():
    batch_size = 2
    seq_len = 3
    channels = 3
    height, width = 224, 224
    
    encoder = VisualEncoder(img_size=224,patch_size=16,in_channels=3,embed_dim=192,num_heads=3,mlp_ratio=2.0,dropout=0.1,num_layers=3,output_dim=512)
    
    # Test with sequence input
    x = torch.randn(batch_size, seq_len, channels, height, width)
    output = encoder(x)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, 512)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test with flat input (batch_size*seq_len, channels, height, width)
    x_flat = torch.randn(batch_size * seq_len, channels, height, width)
    output_flat = encoder(x_flat)
    
    # Check output shape
    expected_flat_shape = (batch_size * seq_len, 512)
    assert output_flat.shape == expected_flat_shape, f"Expected shape {expected_flat_shape}, got {output_flat.shape}"
    
    # Test with different batch size
    batch_size_2 = 4
    x_2 = torch.randn(batch_size_2, seq_len, channels, height, width)
    output_2 = encoder(x_2)
    assert output_2.shape == (batch_size_2, seq_len, 512), f"Failed batch size generalization test"
    
    print("ViT test passed!")

if __name__ == "__main__":
    test_vit_encoder()