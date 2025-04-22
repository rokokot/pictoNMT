# scripts/prepare_eole_data.py
import os
import json
import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_eole_format(input_file, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):     # best see docs, this is very loose scratched up from the brief demos
    
    os.makedirs(output_dir, exist_ok=True) # keep eole data separate
    
    # load propicto json data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {input_file}")
    
    # make splits
    train_val, test = train_test_split(data, test_size=split_ratio[2], random_state=seed)
    train, val = train_test_split(train_val, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=seed)
    
    logger.info(f"Split data into {len(train)} train, {len(val)} val, and {len(test)} test examples")
    
    process_split(train, output_dir, "train")
    process_split(val, output_dir, "valid")
    process_split(test, output_dir, "test")
    
    create_vocabulary(data, output_dir)
    
    logger.info("Data conversion complete")

def process_split(data, output_dir, split_name):
    src_file = os.path.join(output_dir, f"{split_name}.picto")
    with open(src_file, 'w', encoding='utf-8') as f:
        for item in data:
            picto_sequence = ' '.join(map(str, item['pictogram_sequence']))
            f.write(f"{picto_sequence}\n")
    
    tgt_file = os.path.join(output_dir, f"{split_name}.fr")
    with open(tgt_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item['target_text']}\n")
    
    meta_file = os.path.join(output_dir, f"{split_name}.meta.json")
    metadata = []
    for item in data:
        meta_item = {"pictogram_sequence": item['pictogram_sequence'],"pictogram_tokens": item.get('pictogram_tokens', '')}
        metadata.append(meta_item)
    
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Processed {len(data)} examples for {split_name} split")

def create_vocabulary(data, output_dir):
    vocab_dir = os.path.join(output_dir, "../vocabulary")
    os.makedirs(vocab_dir, exist_ok=True)
    
    picto_ids = set()
    for item in data:
        picto_ids.update(item['pictogram_sequence'])
    
    sorted_picto_ids = sorted(picto_ids)
    
    picto_vocab_file = os.path.join(vocab_dir, "picto.vocab")
    with open(picto_vocab_file, 'w', encoding='utf-8') as f:
        f.write("<unk>\n<s>\n</s>\n<pad>\n")    #check eole ref
        for picto_id in sorted_picto_ids:
            f.write(f"{picto_id}\n")
    
    logger.info(f"Created pictogram vocabulary with {len(sorted_picto_ids)} entries")
    
    fr_words = set()
    for item in data:
        words = item['target_text'].split()
        fr_words.update(words)
    
    sorted_fr_words = sorted(fr_words)
    
    fr_vocab_file = os.path.join(vocab_dir, "fr.vocab")
    with open(fr_vocab_file, 'w', encoding='utf-8') as f:
        f.write("<unk>\n<s>\n</s>\n<pad>\n")
        for word in sorted_fr_words:
            f.write(f"{word}\n")
    
    logger.info(f" French vocabulary with {len(sorted_fr_words)} entries")

def main(args):
    logger.info(f"Converting {args.input_file} to Eole format")
    convert_to_eole_format(args.input_file, 
        args.output_dir, split_ratio=args.split_ratio, seed=args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process data to Eole format")
    parser.add_argument('--input_file', type=str, default='data/propicto_base.json', help='Input JSON file')
    parser.add_argument('--output_dir', type=str, default='data/processed',  help='Output directory')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='Train/val/test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)