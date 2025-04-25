# src/pictollms/models/__init__.py
from eole.registry import register_encoder
from pictollms.models.encoders.eole_encoder import PictoEoleEncoder

@register_encoder("picto_encoder")
class RegisteredPictoEncoder(PictoEoleEncoder):
    pass