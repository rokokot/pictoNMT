import torch
import torch.nn as nn

"""

Main embedding encoder for metadata, wip ...

Takes the saved metadata generated from arasaac, outputs dense vectors which should be passed to the vision encoder.

Args:
  - torch.Tensor for the category and type

Outs:
  - torch.Tesnor for so called 'semantic features', i.e. vector representations of encoded metadata

  example for category encoding:

# Example from 2627_fr_meta.json:
{"categories": ["number", "mathematics", "core vocabulary-knowledge", "adjective", "numeral adjective"]  
}
# Becomes: category idx [12, 34, 78, 2, 89]
  

for types:

# Example: "ou" (OR) → type="CONJ" → type_id=7

"""



class SemanticEncoder(nn.Module):
    def __init__(self, category_vocab_size=200, type_vocab_size=10, 
                 embedding_dim=256, output_dim=512):
        super().__init__()
        
        self.category_embedding = nn.Embedding(category_vocab_size + 1, embedding_dim // 2)
        
        self.type_embedding = nn.Embedding(type_vocab_size, embedding_dim // 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, output_dim)
        )
    
    def forward(self, categories, types):
       
        cat_embeds = self.category_embedding(categories)  
        cat_embeds = cat_embeds.mean(dim=2)  # [batch, seq, embed_dim//2] take a look at dimensions here, do they carry over to pipeline
        
        # Embed types for pictos, one per each picto
        type_embeds = self.type_embedding(types)  # [batch, seq, embed_dim//2]
        
        # Concatenate, probably
        combined = torch.cat([cat_embeds, type_embeds], dim=2)  # [batch, seq, embed_dim], same dim issue
        
        output = self.mlp(combined)
        
        return output