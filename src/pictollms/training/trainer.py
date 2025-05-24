# scripts/training/train_fixed.py
"""
Fixed training script for standalone PictoNMT with proper decoder integration
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pictollms.models.complete.pictonmt import PictoNMT, create_model_config
from pictollms.data.dataset import PictoDataset
from pictollms.data.image_processor import ImageProcessor
from pictollms.eval.metrics import evaluate_translations

def setup_logging(output_dir):
    """Set up logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_fixed_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def prepare_data(args, logger):
    """Prepare training data - FIXED VERSION"""
    logger.info("Preparing training data...")
    
    # Check if processed data exists
    processed_file = project_root / "data" / "propicto_base.json"
    
    if not processed_file.exists():
        # Process raw data
        logger.info("Processing raw PropictoOrféo data...")
        from scripts.data_processing.process_propicto import process_propicto_files
        
        input_dir = str(project_root / "data" / "propicto-source")
        output_file = str(processed_file)
        
        process_propicto_files(input_dir, output_file)
        
        if not processed_file.exists():
            raise FileNotFoundError("Failed to create processed data file")
    
    # Load and split data
    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Split data
    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create dataset files
    dataset_dir = project_root / "data" / "training_split"
    dataset_dir.mkdir(exist_ok=True)
    
    # Save splits
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        # Create .picto and .fr files
        with open(dataset_dir / f"{split_name}.picto", 'w') as picto_f, \
             open(dataset_dir / f"{split_name}.fr", 'w') as fr_f:
            
            for item in split_data:
                picto_line = ' '.join(map(str, item['pictogram_sequence']))
                picto_f.write(picto_line + '\n')
                fr_f.write(item['target_text'] + '\n')
        
        # Create metadata file
        metadata = [{'pictogram_sequence': item['pictogram_sequence']} for item in split_data]
        with open(dataset_dir / f"{split_name}.meta.json", 'w') as f:
            json.dump(metadata, f)
    
    return str(dataset_dir)

def create_data_loaders(dataset_dir, tokenizer, args, logger):
    """Create data loaders - FIXED VERSION"""
    logger.info("Creating data loaders...")
    
    # Initialize image processor
    image_processor = ImageProcessor(
        lmdb_path=args.lmdb_path if hasattr(args, 'lmdb_path') else "nonexistent.lmdb",
        resolution=224
    )
    
    # Create datasets
    train_dataset = PictoDataset(
        data_file=os.path.join(dataset_dir, "train"),
        metadata_file=os.path.join(dataset_dir, "train.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = PictoDataset(
        data_file=os.path.join(dataset_dir, "val"),
        metadata_file=os.path.join(dataset_dir, "val.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # FIXED: Improved collate function
    def collate_fn(batch):
        # Extract components
        pictogram_sequences = [item['pictogram_sequence'] for item in batch]
        images = [item['images'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        
        # Pad sequences
        max_picto_len = max(len(seq) for seq in pictogram_sequences)
        max_target_len = max(len(tgt) for tgt in target_ids)
        
        # Pad pictogram sequences and create masks
        padded_picto_seqs = []
        attention_masks = []
        
        for seq in pictogram_sequences:
            padded_seq = seq.tolist() + [0] * (max_picto_len - len(seq))
            mask = [1] * len(seq) + [0] * (max_picto_len - len(seq))
            padded_picto_seqs.append(padded_seq)
            attention_masks.append(mask)
        
        # Pad and stack images
        padded_images = []
        for img in images:
            if img.shape[0] < max_picto_len:
                padding = torch.zeros(max_picto_len - img.shape[0], *img.shape[1:])
                padded_img = torch.cat([img, padding], dim=0)
            else:
                padded_img = img[:max_picto_len]
            padded_images.append(padded_img)
        
        # Pad targets - FIXED: Ensure proper padding for teacher forcing
        padded_targets = []
        for tgt in target_ids:
            tgt_list = tgt.tolist()
            padded_tgt = tgt_list + [tokenizer.pad_token_id] * (max_target_len - len(tgt_list))
            padded_targets.append(padded_tgt[:max_target_len])  # Ensure consistent length
        
        # Create metadata tensors (simplified)
        batch_size = len(batch)
        categories = torch.zeros(batch_size, max_picto_len, 5, dtype=torch.long)
        types = torch.zeros(batch_size, max_picto_len, dtype=torch.long)
        
        return {
            'images': torch.stack(padded_images),
            'categories': categories,
            'types': types,
            'attention_mask': torch.tensor(attention_masks),
            'target_ids': torch.tensor(padded_targets),
            'target_texts': target_texts
        }
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, scheduler, device, logger, epoch):
    """FIXED: Train for one epoch with proper loss computation"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_schema_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    for batch in progress_bar:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # FIXED: Proper forward pass with mode='train'
            outputs = model(batch, mode='train')
            
            # FIXED: Proper loss computation
            loss_dict = model.compute_loss(outputs, batch)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            total_main_loss += loss_dict.get('main_loss', 0).item() if isinstance(loss_dict.get('main_loss'), torch.Tensor) else loss_dict.get('main_loss', 0)
            total_schema_loss += loss_dict.get('schema_loss', 0).item() if isinstance(loss_dict.get('schema_loss'), torch.Tensor) else loss_dict.get('schema_loss', 0)
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'main': f"{loss_dict.get('main_loss', 0):.4f}",
                'schema': f"{loss_dict.get('schema_loss', 0):.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            # Skip this batch
            continue
    
    return {
        'total_loss': total_loss / num_batches,
        'main_loss': total_main_loss / num_batches,
        'schema_loss': total_schema_loss / num_batches
    }

def validate_epoch(model, val_loader, device, tokenizer, logger, epoch):
    """FIXED: Validate for one epoch with proper generation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                # Forward pass for loss
                outputs = model(batch, mode='train')
                loss_dict = model.compute_loss(outputs, batch)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Generate predictions for first few batches
                if batch_idx < 5:  # Only for first 5 batches to save time
                    try:
                        # FIXED: Proper generation call
                        predictions = model.generate(batch, strategy='greedy', tokenizer=tokenizer)
                        
                        # Decode predictions
                        for i, pred in enumerate(predictions):
                            if isinstance(pred, list) and len(pred) > 0:
                                decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
                                all_predictions.append(decoded_pred)
                                
                                if i < len(batch['target_texts']):
                                    all_references.append(batch['target_texts'][i])
                    except Exception as e:
                        logger.warning(f"Generation failed in validation: {e}")
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })
                
            except Exception as e:
                logger.warning(f"Error in validation step: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate metrics if we have predictions
    metrics = {}
    if all_predictions and len(all_predictions) == len(all_references):
        try:
            metrics = evaluate_translations(all_predictions, all_references)
            
            # Log some examples
            logger.info("Validation Examples:")
            for i in range(min(3, len(all_predictions))):
                logger.info(f"  Ref: {all_references[i]}")
                logger.info(f"  Pred: {all_predictions[i]}")
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
    
    return avg_loss, metrics

def main():
    """FIXED: Main training function"""
    parser = argparse.ArgumentParser(description="Train PictoNMT Standalone - Fixed Version")
    
    # Data arguments
    parser.add_argument('--max_length', type=int, default=100, help='Max sequence length')
    parser.add_argument('--lmdb_path', type=str, default='data/cache/images/pictograms.lmdb', help='LMDB path for images')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='Data loader workers')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--visual_layers', type=int, default=6, help='Visual encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Decoder layers')
    
    # Output arguments
    parser.add_argument('--output_dir', default='models/standalone_fixed', help='Output directory')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting PictoNMT Fixed Training")
    print(f"Using device: {device}")
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting training with args: {vars(args)}")
    
    try:
        # Prepare data
        dataset_dir = prepare_data(args, logger)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        
        # Create model
        logger.info("Creating model...")
        config = create_model_config(len(tokenizer))
        
        # Update config with command line args
        config.update({
            'hidden_dim': args.hidden_dim,
            'visual_layers': args.visual_layers,
            'decoder_layers': args.decoder_layers
        })
        
        model = PictoNMT(vocab_size=len(tokenizer), config=config)
        model.pad_token_id = tokenizer.pad_token_id  # FIXED: Set pad token ID
        model.to(device)
        
        # Log model info
        model_size_info = model.get_model_size()
        logger.info(f"Model created: {model_size_info}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(dataset_dir, tokenizer, args, logger)
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # FIXED: Proper scheduler setup
        total_steps = len(train_loader) * args.num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_metrics = {}
        
        logger.info(f"Starting training for {args.num_epochs} epochs...")
        
        for epoch in range(args.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_results = train_epoch(model, train_loader, optimizer, scheduler, device, logger, epoch)
            
            # Validate
            val_loss, val_metrics = validate_epoch(model, val_loader, device, tokenizer, logger, epoch)
            
            # Log results
            logger.info(f"Epoch {epoch+1} Results:")
            logger.info(f"  Train Loss: {train_results['total_loss']:.4f} (Main: {train_results['main_loss']:.4f}, Schema: {train_results['schema_loss']:.4f})")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            if val_metrics:
                logger.info(f"  BLEU: {val_metrics.get('bleu', 0):.2f}")
                logger.info(f"  ROUGE-L: {val_metrics.get('rouge_l', 0):.2f}")
                logger.info(f"  Content Preservation: {val_metrics.get('content_preservation', 0):.2f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_metrics = val_metrics or {}
                logger.info(f"✨ New best model! Val Loss: {val_loss:.4f}")
            
            if (epoch + 1) % args.save_every == 0 or is_best:
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_results': train_results,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics or {},
                    'config': config,
                    'args': vars(args),
                    'tokenizer_name': "flaubert/flaubert_small_cased"
                }
                
                # Save regular checkpoint
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
                
                # Save best model
                if is_best:
                    best_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save(checkpoint, best_path)
                    logger.info(f"💾 Saved best model: {best_path}")
        
        # Training completed
        logger.info(f"🎉 Training completed!")
        logger.info(f"Final Results:")
        logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
        if best_metrics:
            logger.info(f"  Best BLEU: {best_metrics.get('bleu', 0):.2f}")
            logger.info(f"  Best ROUGE-L: {best_metrics.get('rouge_l', 0):.2f}")
        
        # Save final training summary
        summary = {
            'training_completed': True,
            'num_epochs': args.num_epochs,
            'best_val_loss': best_val_loss,
            'best_metrics': best_metrics,
            'model_config': config,
            'training_args': vars(args),
            'model_size_info': model_size_info
        }
        
        summary_path = os.path.join(args.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved: {summary_path}")
        logger.info("✅ Fixed training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()