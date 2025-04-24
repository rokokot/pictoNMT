# pictollms/data/dataset.py
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional, List

class PictoDataset(Dataset):
    def __init__(self, data_file: str, metadata_file: str, image_processor=None, tokenizer=None, max_length: int = 100):
        
        self.source_sequences = self._load_sequences(data_file + ".picto")
        self.target_sequences = self._load_sequences(data_file + ".fr")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _load_sequences(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def __len__(self):
        return len(self.source_sequences)
    
    def __getitem__(self, idx):
        source_sequence = self.source_sequences[idx]
        picto_ids = [int(x) for x in source_sequence.split()]
        target_sequence = self.target_sequences[idx]
        metadata = self.metadata[idx]
        
        item = {'pictogram_sequence': torch.tensor(picto_ids),'target_text': target_sequence,'metadata': metadata}
        
        if self.image_processor is not None:
            item['images'] = self.image_processor.get_batch_images(picto_ids)
        
        if self.tokenizer is not None:
            target_encoding = self.tokenizer(target_sequence,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt')
            
            item['target_ids'] = target_encoding['input_ids'].squeeze()
            item['target_attention_mask'] = target_encoding['attention_mask'].squeeze()
        
        return item