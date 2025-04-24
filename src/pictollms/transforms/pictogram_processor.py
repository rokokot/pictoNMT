# custom_modules/transforms/pictogram_processor.py
import os
import json
import torch
from eole.transforms import Transform
from typing import Dict, List, Optional

class PictogramProcessor(Transform):
        
    def __init__(self, metadata_path: str):
        
        super().__init__()
        self.metadata_path = metadata_path
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.type_map = {'NOUN': 0,'VERB': 1,'ADJ': 2,'ADV': 3,'PREP': 4,'DET': 5,'PRON': 6,'CONJ': 7,'NUM': 8,'UNKNOWN': 9}
    
    def _parse(self, raw_pictogram_sequence: str):
        
        picto_ids = [int(x) for x in raw_pictogram_sequence.split()]
        types = []
        categories = []
        
        for picto_id in picto_ids:
            picto_meta = self.metadata.get(str(picto_id), {})
            
            picto_type = picto_meta.get('type', 'UNKNOWN')
            type_id = self.type_map.get(picto_type, self.type_map['UNKNOWN'])
            types.append(type_id)
            
            picto_categories = picto_meta.get('categories', [])
            cat_list = picto_categories[:5]
            cat_list = cat_list + [''] * (5 - len(cat_list))
            categories.append(cat_list)
        
        processed = {'src': picto_ids,'metadata': {'types': types,'categories': categories}}
        
        return processed
    
    def __call__(self, example: Dict) -> Dict:
        processed = self._parse(example['src'])
        example['src'] = processed['src']
        example['metadata'] = processed['metadata']
        return example