# scripts/build_metadata.py
import os
import json
import argparse
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

from pictollms.data.arasaac_client import ArasaacClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_unique_pictogram_ids(data_file):    # check list format for retrieving uuid
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    unique_picto_ids = set()    # set for pictos
    for item in data:
        unique_picto_ids.update(item['pictogram_sequence'])
    
    return list(unique_picto_ids)

def process_image(image_path, target_size=(224, 224)):    # image helper, check resolution
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def build_image_database(unique_ids, arasaac_client, lmdb_path, resolution=224):
    map_size = 1024 * 1024 * 1024 * 10      # maybe? 10GB
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    count = 0
    with env.begin(write=True) as txn:      # lmdb from docs, depends on how file is saved and passed to encoder
        for picto_id in tqdm(unique_ids, desc="Processing images"):
            img_path = arasaac_client.download_pictogram(picto_id, resolution)

            if img_path:
                img_array = process_image(img_path, (resolution, resolution))

                if img_array is not None:   # check whether this handles extra chars on picture (arasaac)
                    
                    key = f"picto_{picto_id}".encode()
                    value = np.ascontiguousarray(img_array)
                    txn.put(key, value.tobytes())
                    count += 1
    
    logger.info(f"Added {count} images to LMDB database")
    env.close()


def main(args):   # main function to call the metadata components, later used for dataset building
    
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.metadata_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.lmdb_path), exist_ok=True)
    
    client = ArasaacClient(cache_dir=args.cache_dir)
    
    
    logger.info(f"Extracting unique pictogram IDs from {args.data_file}")     # get unique pictogram IDs
    unique_ids = extract_unique_pictogram_ids(args.data_file)
    logger.info(f"Found {len(unique_ids)} unique pictogram IDs")
    
    logger.info("Collecting metadata for pictograms")
    metadata_dict = {}    # keep metadata for easy retrieval

    for picto_id in tqdm(unique_ids, desc="getting metadata"):
        metadata = client.get_pictogram_metadata(picto_id)
        if metadata:
            
              
            processed_metadata = {
                'id': picto_id,
                'keywords': metadata.get('keywords', []),
                'categories': [cat.get('keyword') for cat in metadata.get('categories', [])],
                'type': determine_type(metadata),
                'fr_label': extract_french_label(metadata),
                'fr_alternatives': extract_french_alternatives(metadata)
            }


            metadata_dict[str(picto_id)] = processed_metadata
    
    # Save consolidated metadata
    metadata_file = os.path.join(args.metadata_dir, 'arasaac_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata for {len(metadata_dict)} pictograms to {metadata_file}")
    
    # Build image database
    if args.build_images:
        logger.info(f"Building image database at {args.lmdb_path}")
        build_image_database(unique_ids, client, args.lmdb_path, args.resolution)
    
    logger.info("Metadata and image collection complete")

def determine_type(metadata):
    """Determine grammatical type of pictogram"""
    # Extract categories
    categories = [cat.get('keyword', '').lower() for cat in metadata.get('categories', [])]
    
    # Determine type based on categories
    if any(c in categories for c in ['noun', 'object', 'person']):
        return 'NOUN'
    elif any(c in categories for c in ['verb', 'action']):
        return 'VERB'
    elif any(c in categories for c in ['adjective', 'description']):
        return 'ADJ'
    elif any(c in categories for c in ['adverb']):
        return 'ADV'
    elif any(c in categories for c in ['preposition']):
        return 'PREP'
    else:
        return 'UNKNOWN'

def extract_french_label(metadata):
    """Extract primary French label"""
    keywords = metadata.get('keywords', [])
    for keyword in keywords:
        if keyword.get('language') == 'fr':
            return keyword.get('keyword', '')
    return ''

def extract_french_alternatives(metadata):
    """Extract alternative French labels"""
    keywords = metadata.get('keywords', [])
    alternatives = []
    primary = extract_french_label(metadata)
    
    for keyword in keywords:
        if keyword.get('language') == 'fr':
            word = keyword.get('keyword', '')
            if word and word != primary:
                alternatives.append(word)
    
    return alternatives

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build metadata and image database for pictograms")
    parser.add_argument('--data_file', type=str, default='data/propicto_base.json',
                       help='Path to processed data file')
    parser.add_argument('--cache_dir', type=str, default='data/cache/arasaac',
                       help='Directory for caching API responses')
    parser.add_argument('--metadata_dir', type=str, default='data/metadata',
                       help='Directory for saving metadata')
    parser.add_argument('--lmdb_path', type=str, default='data/cache/images/pictograms.lmdb',
                       help='Path for LMDB image database')
    parser.add_argument('--build_images', action='store_true',
                       help='Whether to build image database')
    parser.add_argument('--resolution', type=int, default=224,
                       help='Resolution for pictogram images')
    
    args = parser.parse_args()
    main(args)