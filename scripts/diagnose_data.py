#!/usr/bin/env python3
import os
import json
import lmdb
import torch
import numpy as np
from pathlib import Path

def check_processed_data():
    """Check if processed data exists and is valid"""
    print("=== Checking Processed Data ===")
    
    data_dir = "data/processed"
    files_to_check = ["train.picto", "train.fr", "train.meta.json"]
    
    for file in files_to_check:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            print(f"✅ {file} exists")
            if file.endswith('.picto'):
                # Check first few lines
                with open(filepath, 'r') as f:
                    lines = [f.readline().strip() for _ in range(3)]
                print(f"   First 3 pictogram sequences:")
                for i, line in enumerate(lines):
                    picto_ids = line.split()[:5]  # Show first 5 IDs
                    print(f"   Line {i+1}: {' '.join(picto_ids)}...")
            elif file.endswith('.fr'):
                with open(filepath, 'r') as f:
                    lines = [f.readline().strip() for _ in range(3)]
                print(f"   First 3 French sentences:")
                for i, line in enumerate(lines):
                    print(f"   Line {i+1}: {line[:50]}...")
        else:
            print(f"❌ {file} missing")

def check_metadata():
    """Check metadata file"""
    print("\n=== Checking Metadata ===")
    
    metadata_file = "data/metadata/arasaac_metadata.json"
    if os.path.exists(metadata_file):
        print(f"✅ Metadata file exists")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"   Total pictograms in metadata: {len(metadata)}")
        
        # Show first few entries
        first_5_keys = list(metadata.keys())[:5]
        print("   First 5 pictogram IDs in metadata:")
        for key in first_5_keys:
            label = metadata[key].get('fr_label', 'N/A')
            print(f"   {key}: {label}")
        
        return metadata
    else:
        print(f"❌ Metadata file missing: {metadata_file}")
        return None

def check_lmdb_cache():
    """Check LMDB image cache"""
    print("\n=== Checking LMDB Image Cache ===")
    
    lmdb_path = "data/cache/images/pictograms.lmdb"
    if os.path.exists(lmdb_path):
        print(f"✅ LMDB database exists: {lmdb_path}")
        try:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                count = 0
                cached_ids = []
                
                for key, value in cursor:
                    count += 1
                    key_str = key.decode()
                    if key_str.startswith('picto_'):
                        picto_id = key_str.replace('picto_', '')
                        cached_ids.append(picto_id)
                    
                    # Check first image
                    if count == 1:
                        print(f"   First key: {key_str}")
                        try:
                            img_array = np.frombuffer(value, dtype=np.float32)
                            expected_size = 224 * 224 * 3  # Height * Width * Channels
                            print(f"   Image array size: {len(img_array)} (expected: {expected_size})")
                            if len(img_array) == expected_size:
                                print(f"   ✅ Image data looks correct")
                                img_reshaped = img_array.reshape(224, 224, 3)
                                print(f"   Image shape: {img_reshaped.shape}")
                                print(f"   Value range: [{img_reshaped.min():.3f}, {img_reshaped.max():.3f}]")
                            else:
                                print(f"   ❌ Unexpected image size")
                        except Exception as e:
                            print(f"   ❌ Error reading image: {e}")
                
                print(f"   Total entries in LMDB: {count}")
                print(f"   Sample cached pictogram IDs: {cached_ids[:10]}")
                
                env.close()
                return cached_ids
        except Exception as e:
            print(f"❌ Error accessing LMDB: {e}")
            return []
    else:
        print(f"❌ LMDB database missing: {lmdb_path}")
        return []

def check_id_consistency():
    """Check if pictogram IDs in data match those in cache"""
    print("\n=== Checking ID Consistency ===")
    
    # Get pictogram IDs from processed data
    data_ids = set()
    data_file = "data/processed/train.picto"
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= 100:  # Just check first 100 lines
                    break
                ids = line.strip().split()
                data_ids.update(ids)
        
        print(f"   Unique pictogram IDs in first 100 training examples: {len(data_ids)}")
        sample_data_ids = list(data_ids)[:10]
        print(f"   Sample data IDs: {sample_data_ids}")
    else:
        print(f"❌ Training data file not found")
        return
    
    # Get cached IDs
    cached_ids = check_lmdb_cache()
    if cached_ids:
        cached_ids_set = set(cached_ids)
        
        # Find intersection
        common_ids = data_ids.intersection(cached_ids_set)
        missing_from_cache = data_ids - cached_ids_set
        
        print(f"   IDs in both data and cache: {len(common_ids)}")
        print(f"   IDs missing from cache: {len(missing_from_cache)}")
        
        if missing_from_cache:
            print(f"   Sample missing IDs: {list(missing_from_cache)[:10]}")
            
        # Test loading a few images
        if common_ids:
            print("\n   Testing image loading for common IDs:")
            from pictollms.data.image_processor import ImageProcessor
            
            processor = ImageProcessor(lmdb_path="data/cache/images/pictograms.lmdb")
            test_ids = list(common_ids)[:3]
            
            for picto_id in test_ids:
                image = processor.get_image(int(picto_id))
                if image is not None:
                    print(f"   ✅ Successfully loaded image for ID {picto_id}: shape {image.shape}")
                else:
                    print(f"   ❌ Failed to load image for ID {picto_id}")

def test_dataset_loading():
    """Test loading a small batch from the dataset"""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        from transformers import AutoTokenizer
        from pictollms.data.dataset import PictoDataset
        from pictollms.data.image_processor import ImageProcessor
        
        # Set up tokenizer and image processor
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        image_processor = ImageProcessor(lmdb_path="data/cache/images/pictograms.lmdb")
        
        # Create dataset
        dataset = PictoDataset(
            data_file="data/processed/train",
            metadata_file="data/processed/train.meta.json",
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=50
        )
        
        print(f"   Dataset size: {len(dataset)}")
        
        # Test loading first few items
        for i in range(min(3, len(dataset))):
            try:
                item = dataset[i]
                print(f"   Item {i}:")
                print(f"     Pictogram sequence length: {len(item['pictogram_sequence'])}")
                print(f"     Images shape: {item.get('images', 'Missing').shape if 'images' in item else 'Missing'}")
                print(f"     Target text: {item['target_text'][:50]}...")
            except Exception as e:
                print(f"   ❌ Failed to load item {i}: {e}")
                
    except Exception as e:
        print(f"   ❌ Error setting up dataset test: {e}")

def main():
    print("PictoNMT Data Diagnosis")
    print("=" * 50)
    
    check_processed_data()
    metadata = check_metadata()
    cached_ids = check_lmdb_cache()
    check_id_consistency()
    test_dataset_loading()
    
    print("\n" + "=" * 50)
    print("Diagnosis complete!")
    
    # Provide recommendations
    print("\nRecommendations:")
    if not os.path.exists("data/processed/train.picto"):
        print("1. Run: python scripts/data_processing/eole_data.py")
    
    if not os.path.exists("data/metadata/arasaac_metadata.json"):
        print("2. Run: python scripts/data_processing/build_metadata.py --data_file data/propicto_base.json")
    
    if not os.path.exists("data/cache/images/pictograms.lmdb"):
        print("3. Run: python scripts/data_processing/build_metadata.py --data_file data/propicto_base.json --build_images")

if __name__ == "__main__":
    main()