# src/pictollms/data/metadata_processor.py
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MetadataProcessor:
    """Process pictogram metadata for semantic encoding - FIXED VERSION"""
    
    def __init__(self, metadata_file: str = None):
        """
        Initialize metadata processor
        
        Args:
            metadata_file: Path to ARASAAC metadata JSON file
        """
        self.metadata = {}
        self.category_map = {}
        self.type_map = {
            'NOUN': 1,
            'VERB': 2,
            'ADJ': 3,
            'ADV': 4,
            'PREP': 5,
            'DET': 6,
            'PRON': 7,
            'CONJ': 8,
            'NUM': 9,
            'UNKNOWN': 0
        }
        
        if metadata_file and os.path.exists(metadata_file):
            self._load_metadata(metadata_file)
        else:
            logger.warning(f"Metadata file not found: {metadata_file}. Using default mappings.")
            self._create_default_mappings()
    
    def _load_metadata(self, metadata_file: str):
        """Load metadata from file"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded metadata for {len(self.metadata)} pictograms")
            self._create_category_map()
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self._create_default_mappings()
    
    def _create_category_map(self):
        """Create mapping from categories to indices"""
        all_categories = set()
        for picto_id, info in self.metadata.items():
            categories = info.get('categories', [])
            if isinstance(categories, list):
                all_categories.update(categories)
        
        # Create mapping with common categories
        common_categories = [
            'noun', 'verb', 'adjective', 'adverb', 'preposition',
            'person', 'action', 'object', 'place', 'time',
            'food', 'animal', 'color', 'number', 'emotion',
            'body', 'clothing', 'transport', 'house', 'school',
            'family', 'health', 'sport', 'music', 'technology'
        ]
        
        # Add common categories first
        self.category_map = {cat: i+1 for i, cat in enumerate(common_categories)}
        
        # Add remaining categories from data
        next_idx = len(common_categories) + 1
        for cat in sorted(all_categories):
            if cat.lower() not in self.category_map:
                self.category_map[cat.lower()] = next_idx
                next_idx += 1
        
        logger.info(f"Created category mapping with {len(self.category_map)} categories")
    
    def _create_default_mappings(self):
        """Create default mappings when metadata is not available"""
        # Default categories
        default_categories = [
            'noun', 'verb', 'adjective', 'adverb', 'preposition',
            'person', 'action', 'object', 'place', 'time'
        ]
        
        self.category_map = {cat: i+1 for i, cat in enumerate(default_categories)}
        logger.info("Created default category mappings")
    
    def get_metadata_features(self, picto_id: int) -> Dict[str, torch.Tensor]:
        """
        Get metadata features for a pictogram - FIXED VERSION
        
        Args:
            picto_id: Pictogram ID
            
        Returns:
            Dictionary of metadata features as tensors
        """
        # Get metadata for pictogram or use defaults
        picto_info = self.metadata.get(str(picto_id), {})
        
        # Extract and process categories
        categories = picto_info.get('categories', [])
        if not isinstance(categories, list):
            categories = []
        
        # Map categories to indices
        category_indices = []
        for cat in categories[:5]:  # Limit to 5 categories
            cat_lower = cat.lower() if isinstance(cat, str) else str(cat).lower()
            idx = self.category_map.get(cat_lower, 0)  # 0 for unknown
            category_indices.append(idx)
        
        # Pad to exactly 5 categories
        while len(category_indices) < 5:
            category_indices.append(0)
        
        # Extract and process type
        picto_type = picto_info.get('type', 'UNKNOWN')
        if not isinstance(picto_type, str):
            picto_type = 'UNKNOWN'
        
        type_idx = self.type_map.get(picto_type.upper(), 0)
        
        return {
            'categories': torch.tensor(category_indices, dtype=torch.long),
            'type': torch.tensor(type_idx, dtype=torch.long),
            'has_metadata': torch.tensor(1 if str(picto_id) in self.metadata else 0, dtype=torch.long)
        }
    
    def batch_process(self, picto_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of pictogram IDs - FIXED VERSION
        
        Args:
            picto_ids: List of pictogram IDs
            
        Returns:
            Dictionary of batched metadata features
        """
        batch_categories = []
        batch_types = []
        batch_has_metadata = []
        
        for picto_id in picto_ids:
            features = self.get_metadata_features(picto_id)
            batch_categories.append(features['categories'])
            batch_types.append(features['type'])
            batch_has_metadata.append(features['has_metadata'])
        
        return {
            'categories': torch.stack(batch_categories),
            'types': torch.stack(batch_types),
            'has_metadata': torch.stack(batch_has_metadata)
        }
    
    def get_category_info(self) -> Dict[str, int]:
        """Get category mapping information"""
        return self.category_map.copy()
    
    def get_type_info(self) -> Dict[str, int]:
        """Get type mapping information"""
        return self.type_map.copy()
    
    def get_stats(self) -> Dict[str, any]:
        """Get metadata statistics"""
        if not self.metadata:
            return {
                'total_pictograms': 0,
                'categories_available': len(self.category_map),
                'types_available': len(self.type_map)
            }
        
        # Analyze metadata
        total_with_categories = 0
        total_with_types = 0
        category_counts = {}
        type_counts = {}
        
        for picto_id, info in self.metadata.items():
            categories = info.get('categories', [])
            if categories:
                total_with_categories += 1
                for cat in categories:
                    cat_lower = cat.lower() if isinstance(cat, str) else str(cat).lower()
                    category_counts[cat_lower] = category_counts.get(cat_lower, 0) + 1
            
            picto_type = info.get('type', 'UNKNOWN')
            if picto_type and picto_type != 'UNKNOWN':
                total_with_types += 1
                type_counts[picto_type] = type_counts.get(picto_type, 0) + 1
        
        return {
            'total_pictograms': len(self.metadata),
            'pictograms_with_categories': total_with_categories,
            'pictograms_with_types': total_with_types,
            'categories_available': len(self.category_map),
            'types_available': len(self.type_map),
            'most_common_categories': sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'most_common_types': sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }


# FIXED: Update the dataset to use metadata processor
class PictoDatasetFixed(torch.utils.data.Dataset):
    """Fixed dataset that properly integrates metadata"""
    
    def __init__(self, data_file: str, metadata_file: str, image_processor=None, 
                 tokenizer=None, max_length: int = 100, arasaac_metadata_file: str = None):
        
        self.source_sequences = self._load_sequences(data_file + ".picto")
        self.target_sequences = self._load_sequences(data_file + ".fr")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.sequence_metadata = json.load(f)
        
        # FIXED: Initialize metadata processor
        self.metadata_processor = MetadataProcessor(arasaac_metadata_file)
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Log metadata stats
        stats = self.metadata_processor.get_stats()
        logger.info(f"Dataset metadata stats: {stats}")
    
    def _load_sequences(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def __len__(self):
        return len(self.source_sequences)
    
    def __getitem__(self, idx):
        source_sequence = self.source_sequences[idx]
        picto_ids = [int(x) for x in source_sequence.split()]
        target_sequence = self.target_sequences[idx]
        sequence_metadata = self.sequence_metadata[idx]
        
        item = {
            'pictogram_sequence': torch.tensor(picto_ids),
            'target_text': target_sequence,
            'sequence_metadata': sequence_metadata
        }
        
        # FIXED: Process metadata for each pictogram
        picto_metadata = self.metadata_processor.batch_process(picto_ids)
        item.update(picto_metadata)
        
        # Process images
        if self.image_processor is not None:
            item['images'] = self.image_processor.get_batch_images(picto_ids)
        else:
            # Create dummy images if no processor
            item['images'] = torch.zeros(len(picto_ids), 3, 224, 224)
        
        # Process target text
        if self.tokenizer is not None:
            target_encoding = self.tokenizer(
                target_sequence,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item['target_ids'] = target_encoding['input_ids'].squeeze()
            item['target_attention_mask'] = target_encoding['attention_mask'].squeeze()
        
        return item