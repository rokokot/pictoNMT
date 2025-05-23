import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pictollms.data.dataset import PictoDataset
from pictollms.data.image_processor import ImageProcessor
from pictollms.models.encoders.eole_encoder import PictoEoleEncoder
from pictollms.models.schema.schema_inducer import SchemaInducer
from pictollms.models.schema.beam_search import CAsiBeamSearch
from pictollms.training.trainer import PictoTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(args):
    """Set up the complete model with all components"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")    # we can play around with the tokenizer but its not clear what effect it has wrt evaluation
    
    # Create encoder
    encoder = PictoEoleEncoder()
    
    # Create schema inducer
    schema_inducer = SchemaInducer(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.schema_layers
    )
    
    # Create CASI beam search
    beam_search = CAsiBeamSearch(
        beam_size=args.beam_size,
        max_length=args.max_length,
        length_penalty=args.length_penalty,
        schema_weight=args.schema_weight
    )
    
    return encoder, schema_inducer, beam_search, tokenizer

def setup_data(args, tokenizer):
    """Set up data loaders"""
    
    # Set up image processor
    image_processor = ImageProcessor(
        lmdb_path=args.lmdb_path,
        resolution=args.image_resolution
    )
    
    # Create datasets
    train_dataset = PictoDataset(
        data_file=os.path.join(args.data_dir, "train"),
        metadata_file=os.path.join(args.data_dir, "train.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = PictoDataset(
        data_file=os.path.join(args.data_dir, "valid"),
        metadata_file=os.path.join(args.data_dir, "valid.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    return train_dataset, val_dataset

def main(args):
    """Main training function"""
    logger.info("Setting up PictoNMT training")
    
    # Set up model components
    encoder, schema_inducer, beam_search, tokenizer = setup_model(args)
    
    # Set up data
    train_dataset, val_dataset = setup_data(args, tokenizer)
    
    # Create trainer
    trainer = PictoTrainer(
        encoder=encoder,
        schema_inducer=schema_inducer,
        beam_search=beam_search,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=args
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train complete PictoNMT system")
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--schema_layers', type=int, default=3)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--length_penalty', type=float, default=0.6)
    parser.add_argument('--schema_weight', type=float, default=0.3)
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--lmdb_path', type=str, default='data/cache/images/pictograms.lmdb')
    parser.add_argument('--image_resolution', type=int, default=224)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='models/complete')
    
    args = parser.parse_args()
    main(args)