try:
    from pictollms.models.encoders.visual_encoder import VisualEncoder
    from pictollms.models.encoders.semantic_encoder import SemanticEncoder  
    from pictollms.models.encoders.dual_path_encoder import DualPathEncoder
    
    __all__ = ['VisualEncoder', 'SemanticEncoder', 'DualPathEncoder']
    print("Core encoders loaded successfully")
    
except ImportError as e:
    print(f"Error loading core encoders: {e}")
    __all__ = []

try:
    from pictollms.models.encoders.eole_encoder import PictoEoleEncoder
    __all__.append('PictoEoleEncoder')
    print("PictoEoleEncoder loaded successfully")
    
    try:
        import eole.encoders
        eole.encoders.str2enc["picto_dual_path"] = PictoEoleEncoder
        print("PictoEoleEncoder registered in Eole's str2enc dictionary")
        print(f"Available Eole encoders: {list(eole.encoders.str2enc.keys())}")
    except Exception as e:
        print(f"Could not register with Eole (triton issue): {e}")
        print("   PictoEoleEncoder is still available for manual use")
        
except ImportError as e:
    print(f"PictoEoleEncoder not available: {e}")

print(f"Available models: {__all__}")