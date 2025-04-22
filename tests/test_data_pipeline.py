# scripts/test_data_pipeline.py
import os
import argparse
import torch
from transformers import AutoTokenizer
import logging
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pictollms.data.arasaac_client import ArasaacClient
from pictollms.data.image_processor import ImageProcessor
from pictollms.data.dataset import PictoDataset

def test_arasaac_client(args):
    
    logger.info("Testing ARASAAC client")
    client = ArasaacClient(cache_dir=args.cache_dir)
    
    logger.info("Testing metadata retrieval")
    test_ids = [11317, 2627, 37406]
    
    for picto_id in test_ids:
        metadata = client.get_pictogram_metadata(picto_id)
        if metadata:
            logger.info(f"Successfully retrieved metadata for pictogram {picto_id}")
            logger.info(f"  Keywords: {metadata.get('keywords', [])[:3]}")
        else:
            logger.warning(f"Failed to retrieve metadata for pictogram {picto_id}")
    
    logger.info("Testing image download")
    for picto_id in test_ids:
        image_path = client.download_pictogram(picto_id, resolution=224)
        if image_path:
            logger.info(f"Successfully downloaded image for pictogram {picto_id}")
            logger.info(f"  Image path: {image_path}")
        else:
            logger.warning(f"Failed to download image for pictogram {picto_id}")

def test_image_processor(args):
    logger.info("Testing image processor")
    processor = ImageProcessor(lmdb_path=args.lmdb_path)
    logger.info("Testing image retrieval")
    test_ids = [11317, 2627, 37406]
    
    for picto_id in test_ids:
        image = processor.get_image(picto_id)
        if image is not None:
            logger.info(f"Successfully retrieved image for pictogram {picto_id}")
            logger.info(f"  Image shape: {image.shape}")
        else:
            logger.warning(f"Failed to retrieve image for pictogram {picto_id}")
    
    logger.info("Testing batch image retrieval")
    batch_images = processor.get_batch_images(test_ids)
    logger.info(f"Batch images shape: {batch_images.shape}")

def test_dataset(args):
    
    logger.info("Testing dataset")
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")    # load a small french model
    processor = ImageProcessor(lmdb_path=args.lmdb_path)
    
    dataset = PictoDataset(data_file=os.path.join(args.data_dir, "train"), metadata_file=os.path.join(args.data_dir, "train.meta.json"), image_processor=processor, tokenizer=tokenizer, max_length=100)
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        logger.info(f"Example {i+1}:")
        logger.info(f"  Pictogram sequence: {item['pictogram_sequence']}")
        logger.info(f"  Target text: {item['target_text']}")
        logger.info(f"  Images shape: {item['images'].shape}")
        logger.info(f"  Target IDs shape: {item['target_ids'].shape}")

def main(args):
    if args.test_client:
        test_arasaac_client(args)
    
    if args.test_processor:
        test_image_processor(args)
    
    if args.test_dataset:
        test_dataset(args)
    
    logger.info("Data pipeline testing complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data pipeline")
    parser.add_argument('--cache_dir', type=str, default='data/cache/arasaac',
                       help='Directory for caching ARASAAC data')
    parser.add_argument('--lmdb_path', type=str, default='data/cache/images/pictograms.lmdb',
                       help='Path to LMDB image database')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--test_client', action='store_true',
                       help='Test ARASAAC client')
    parser.add_argument('--test_processor', action='store_true',
                       help='Test image processor')
  
    parser.add_argument('--test_processor', action='store_true',
                      help='Test image processor')
    parser.add_argument('--test_dataset', action='store_true',
                        help='Test dataset')
    parser.add_argument('--test_all', action='store_true',
                        help='Test all components')
    
    args = parser.parse_args()
    
    if args.test_all:
        args.test_client = True
        args.test_processor = True
        args.test_dataset = True
    
    main(args)