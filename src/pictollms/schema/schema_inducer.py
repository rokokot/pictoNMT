import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

"""
Main semantics module;


"""


class StructureAnalyzer(nn.Module):
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True)

        self.structure_encoder = nn.TransformerEncoder(encoder_layer, num_layers)     # see docs
        
        self.structure_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 10)  # 10 structure types
        )
        
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, encoder_outputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        creates a representation of the structure of encoded pictogram sequence
        
        Args:
            - encoder_outputs: [batch_size, seq_len, hidden_size]
            - attention_mask: [batch_size, seq_len]
        
        Returns:
            -   dict with structure analysis results
        """
        structure_repr = self.structure_encoder(encoder_outputs, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        if attention_mask is not None:
            # we touch on mean pooling elsewhere, 
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(structure_repr)
            sum_repr = (structure_repr * mask_expanded).sum(dim=1)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            global_repr = sum_repr / seq_lengths
        else:
            global_repr = structure_repr.mean(dim=1)
        
        structure_logits = self.structure_classifier(global_repr)
        structure_type = torch.argmax(structure_logits, dim=1)
        complexity_score = self.complexity_estimator(global_repr)
        
        return {
            'structure_repr': structure_repr,
            'global_repr': global_repr,
            'structure_logits': structure_logits,
            'structure_type': structure_type,
            'complexity_score': complexity_score
        }

class FunctionalPredictor(nn.Module):
    """
    predicts functional words needed for French

    looks at categories defined in the encoders, DET, PREP, VBD, CONJ
    
    """
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        
        self.determiner_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4)  # [le, la, les, un, une, des, etc.]
        )
        
        self.preposition_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 6)  # [à, de, dans, sur, avec, pour]
        )
        
        self.verb_auxiliary_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # [être, avoir, none]
        )
        
        self.conjunction_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4)  # [et, ou, mais, car]
        )
    
    def forward(self, structure_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        structure_repr = structure_outputs['structure_repr']
        global_repr = structure_outputs['global_repr']
        
        determiner_logits = self.determiner_predictor(structure_repr)
        preposition_logits = self.preposition_predictor(structure_repr)
        auxiliary_logits = self.verb_auxiliary_predictor(global_repr)
        conjunction_logits = self.conjunction_predictor(global_repr)
        
        return {
            'determiner_logits': determiner_logits,
            'preposition_logits': preposition_logits,
            'auxiliary_logits': auxiliary_logits,
            'conjunction_logits': conjunction_logits
        }

class SchemaInducer(nn.Module):
    
    """
    Main schema induction module making use of functional word prediction
    
    """
    
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        
        self.structure_analyzer = StructureAnalyzer(hidden_size, num_heads, num_layers)
        self.functional_predictor = FunctionalPredictor(hidden_size)
        
        self.schema_enhancer = nn.Sequential(nn.Linear(hidden_size, hidden_size),nn.LayerNorm(hidden_size),nn.ReLU(),nn.Dropout(0.1),nn.Linear(hidden_size, hidden_size))
    
    def forward(self, encoder_outputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        structure_info = self.structure_analyzer(encoder_outputs, attention_mask)
        functional_info = self.functional_predictor(structure_info)
        enhanced_repr = self.schema_enhancer(structure_info['structure_repr'])
        schema = {
            **structure_info,
            **functional_info,
            'enhanced_repr': enhanced_repr,
            'schema_ready': True
        }
        
        return schema