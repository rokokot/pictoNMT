"""
PictoNMT Models Package - Standalone Implementation
No Eole dependencies
"""

__all__ = []

# Core encoders
try:
    from pictollms.models.encoders.visual_encoder import VisualEncoder
    from pictollms.models.encoders.semantic_encoder import SemanticEncoder  
    from pictollms.models.encoders.dual_path_encoder import DualPathEncoder
    
    __all__.extend(['VisualEncoder', 'SemanticEncoder', 'DualPathEncoder'])
    print("✅ Standalone encoders loaded")
    
except ImportError as e:
    print(f"❌ Error loading encoders: {e}")

# Decoder
try:
    from pictollms.models.decoders.transformer_decoder import TransformerDecoder
    __all__.append('TransformerDecoder')
    print("✅ Transformer decoder loaded")
    
except ImportError as e:
    print(f"❌ Error loading decoder: {e}")

# Schema components
try:
    from pictollms.models.schema.schema_inducer import SchemaInducer
    __all__.append('SchemaInducer')
    print("✅ Schema inducer loaded")
    
except ImportError as e:
    print(f"❌ Error loading schema inducer: {e}")

# Complete model
try:
    from pictollms.models.complete.pictonmt import PictoNMT
    __all__.append('PictoNMT')
    print("✅ Complete PictoNMT model loaded")
    
except ImportError as e:
    print(f"❌ Error loading complete model: {e}")

print(f"📦 Available standalone models: {__all__}")
