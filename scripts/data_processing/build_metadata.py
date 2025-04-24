# scripts/build_metadata.py
import os
import json
import argparse
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import sys
import time
from datetime import datetime

from pictollms.data.arasaac_client import ArasaacClient


def setup_logging(log_dir='./logs', log_to_console=True):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"build_metadata_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_unique_pictogram_ids(data_file):    # check list format for retrieving uuid
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    unique_picto_ids = set()    # set for pictos
    for item in data:
        unique_picto_ids.update(item['pictogram_sequence'])
    
    return list(unique_picto_ids)


def process_image(image_path, target_size=(224, 224)):    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def build_image_database(unique_ids, arasaac_client, lmdb_path, target_resolution=224, source_resolution=300):
    map_size = 1024 * 1024 * 1024 * 10  # 10GB
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    count = 0
    failed_count = 0

    with env.begin(write=True) as txn:
        for i, picto_id in enumerate(tqdm(unique_ids, desc="Processing images")):
            # delay every 10 items to avoid overwhelming the server
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)
                
            img_path = arasaac_client.download_pictogram(picto_id, source_resolution)

            if img_path:
                try:
                    # Resize n
                    img_array = process_image(img_path, (target_resolution, target_resolution))

                    if img_array is not None:
                        key = f"picto_{picto_id}".encode()
                        value = np.ascontiguousarray(img_array)
                        txn.put(key, value.tobytes())
                        count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to process image for pictogram {picto_id}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error processing image for pictogram {picto_id}: {e}")
            else:
                failed_count += 1
                logger.warning(f"Failed to download image for pictogram {picto_id}")
            
        if i > 0 and i % 100 == 0:
            logger.info(f"Processed {i}/{len(unique_ids)} images. Success: {count}, Failed: {failed_count}")
    
    logger.info(f"Added {count} images to LMDB database. Failed: {failed_count}")
    env.close()


def main(args):
    log_file = setup_logging(args.log_dir, not args.quiet)
    logger.info(f'Logging to file: {log_file}')
    
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.metadata_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.lmdb_path), exist_ok=True)
    
    client = ArasaacClient(cache_dir=args.cache_dir)
    
    source_resolution = 300
    
    logger.info(f"Extracting unique pictogram IDs from {args.data_file}")
    unique_ids = extract_unique_pictogram_ids(args.data_file)
    logger.info(f"Found {len(unique_ids)} unique pictogram IDs")
    
    if args.build_images:
        test_picto_id = 31410
        logger.info(f'Testing image download with ID {test_picto_id} at resolution {source_resolution}px')
        test_image_path = client.download_pictogram(test_picto_id, source_resolution)
        if test_image_path:
            logger.info(f"Successfully downloaded test image to {test_image_path}")
            test_resized = process_image(test_image_path, (args.resolution, args.resolution))
            if test_resized is not None:
                logger.info(f"Successfully resized test image to {args.resolution}x{args.resolution}")
            else:
                logger.error("Failed to resize test image")
                if not args.force:
                    logger.warning("Image resizing test failed. Continuing with metadata only.")
                    args.build_images = False
        else:
            logger.error(f"Failed to download test image")
            
            if not args.force:
                logger.warning("Image download test failed. Continuing with metadata only.")
                args.build_images = False

    logger.info("Collecting metadata for pictograms")
    metadata_dict = {}

    for picto_id in tqdm(unique_ids, desc="Getting metadata"):
        metadata = client.get_pictogram_metadata(picto_id)
        if metadata:
            processed_metadata = {
                'id': picto_id,
                'keywords': metadata.get('keywords', []),
                'categories': metadata.get('categories', []),
                'type': determine_type(metadata),
                'fr_label': extract_french_label(metadata),
                'fr_alternatives': extract_french_alternatives(metadata),
                'synsets': metadata.get('synsets', []),  
                'tags': metadata.get('tags', []),        
                'properties': {
                    'schematic': metadata.get('schematic', False),
                    'sex': metadata.get('sex', False),
                    'violence': metadata.get('violence', False),
                    'downloads': metadata.get('downloads', 0),
                    'desc': metadata.get('desc', '')
                }
            }
            metadata_dict[str(picto_id)] = processed_metadata
    
    metadata_file = os.path.join(args.metadata_dir, 'arasaac_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata for {len(metadata_dict)} pictograms to {metadata_file}")
    
    if args.build_images:
        logger.info(f"Building image database at {args.lmdb_path}")
        logger.info(f"Downloading at {source_resolution}px and resizing to {args.resolution}px")
        build_image_database(unique_ids, client, args.lmdb_path, args.resolution, source_resolution)
    
    logger.info("Metadata and images processing complete")

def determine_type(metadata):
    # First check categories
    categories = metadata.get('categories', [])
    categories_lower = [c.lower() for c in categories]
    
    if any(c in categories_lower for c in ['noun', 'object', 'person']):
        return 'NOUN'
    elif any(c in categories_lower for c in ['verb', 'action']):
        return 'VERB'
    elif any(c in categories_lower for c in ['adjective', 'description']):
        return 'ADJ'
    elif any(c in categories_lower for c in ['adverb']):
        return 'ADV'
    elif any(c in categories_lower for c in ['preposition']):
        return 'PREP'
    
    tags = metadata.get('tags', [])
    tags_lower = [t.lower() for t in tags]
    
    if 'expression' in tags_lower:
        return 'ADV' 
    
    return 'UNKNOWN'

def extract_french_label(metadata):
    keywords = metadata.get('keywords', [])
    for keyword in keywords:
        if isinstance(keyword, dict) and keyword.get('keyword'):
            return keyword.get('keyword', '')
    return ''

def extract_french_alternatives(metadata):
    keywords = metadata.get('keywords', [])
    alternatives = []
    primary = extract_french_label(metadata)
    
    for keyword in keywords:
        if isinstance(keyword, dict) and keyword.get('keyword'):
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
                       help='Target resolution for pictogram images')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for log files')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output (logs still saved to file)')
    parser.add_argument('--force', action='store_true',
                       help='Force continue even if test image download fails')
    
    args = parser.parse_args()
    main(args)