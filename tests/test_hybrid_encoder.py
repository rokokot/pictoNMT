# tests/test_hybrid_encoder.py
import torch
from pictollms.models.encoders.dual_path_encoder import DualPathEncoder

"""

unit test for the custom encoder module. Look at outs..
testing baseline training without extra custom modules, for now

"""


def test_dual_path_encoder():
    batch_size = 2
    seq_len = 3
    channels = 3
    height, width = 224, 224
    max_categories = 5
    
    images = torch.randn(batch_size, seq_len, channels, height, width)
    categories = torch.randint(0, 100, (batch_size, seq_len, max_categories))
    types = torch.randint(0, 10, (batch_size, seq_len))
    
    visual_config = {'img_size': 224,'patch_size': 16,'in_channels': 3,'embed_dim': 192,'num_heads': 3,'mlp_ratio': 2.0,'dropout': 0.1,'num_layers': 3,'output_dim': 512}
    
    semantic_config = {
        'category_vocab_size': 200,
        'type_vocab_size': 10,
        'embedding_dim': 256,
        'output_dim': 512
    }
    
    fusion_config = {
        'input_dim': visual_config['output_dim'] + semantic_config['output_dim'],
        'hidden_dim': 512,
        'output_dim': 512,
        'dropout': 0.1
    }
    
    encoder = DualPathEncoder(
        visual_config=visual_config,
        semantic_config=semantic_config,
        fusion_config=fusion_config
    )
    
    output = encoder(images, categories, types)
    
    expected_shape = (batch_size, seq_len, fusion_config['output_dim'])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    positions = torch.randint(0, 100, (batch_size, seq_len))
    output_with_positions = encoder(images, categories, types, positions)
    assert output_with_positions.shape == expected_shape, "Output shape changed with custom positions"
    
    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"Dual-path encoder parameter count: {param_count:,}")
    
    dummy_loss = output.sum()
    dummy_loss.backward()
    
    has_gradients = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert has_gradients, "Some parameters did not receive gradients"
    
    print("Dual-path encoder tests passed!")

if __name__ == "__main__":
    test_dual_path_encoder()