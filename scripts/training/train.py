import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import time
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pictollms.models.complete.pictonmt import PictoNMT, create_model_config
from pictollms.data.dataset import PictoDataset
from pictollms.data.image_processor import ImageProcessor
from pictollms.eval.metrics import evaluate_translations

class StandaloneTrainer:
    """Simplified, efficient trainer for PictoNMT"""
    
    def __init__(self, model, tokenizer, train_dataset, val_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store pad token id in model for loss computation
        self.model.pad_token_id = tokenizer.pad_token_id
        
        # Set up data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        total_steps = len(self.train_loader) * config.num_epochs
        warmup_steps = int(total_steps * 0.1)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )
        
        # Set up logging
        self._setup_logging(config.output_dir)
        
        # Track best performance
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
    def _setup_logging(self, output_dir):
        """Set up logging"""
        os.makedirs(output_dir, exist_ok=True)
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")
    
    def _collate_fn(self, batch):
        """Collate function for batching"""
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
        
        # Pad targets
        padded_targets = []
        for tgt in target_ids:
            padded_tgt = tgt.tolist() + [self.tokenizer.pad_token_id] * (max_target_len - len(tgt))
            padded_targets.append(padded_tgt)
        
        # Create metadata (simplified - use zeros for now)
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
    
    def train(self):
        """Main training loop"""
        self.logger.info("PictoNMT training")
        self.logger.info(f"Model parameters: {self.model.get_model_size()}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f" samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            self.logger.info(f"{'='*60}")
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            # Log results
            self.logger.info(f"Epoch {epoch+1} Results:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}")
            if val_metrics:
                self.logger.info(f"  BLEU: {val_metrics.get('bleu', 0):.2f}")
                self.logger.info(f"  ROUGE-L: {val_metrics.get('rouge_l', 0):.2f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics or {}
                self._save_checkpoint('best_model.pt', epoch, val_loss, val_metrics)
                self.logger.info("best model saved!")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss, val_metrics)
        
        self.logger.info(f"Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best metrics: {self.best_metrics}")
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(batch, mode='train')
            
            loss_dict = self.model.compute_loss(outputs, batch)
            loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass for loss
                outputs = self.model(batch, mode='train')
                loss_dict = self.model.compute_loss(outputs, batch)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Generate predictions for evaluation (on subset of batches)
                if batch_idx < 10:  # Only evaluate first 10 batches for speed
                    predictions = self.model.generate(
                        batch, 
                        strategy='casi', 
                        tokenizer=self.tokenizer
                    )
                    
                    # Decode predictions
                    decoded_preds = [
                        self.tokenizer.decode(pred, skip_special_tokens=True) 
                        for pred in predictions
                    ]
                    
                    all_predictions.extend(decoded_preds)
                    all_references.extend(batch['target_texts'])
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })
        
        avg_loss = total_loss / num_batches
        
        metrics = None
        if all_predictions:
            metrics = evaluate_translations(all_predictions, all_references)
            
            # Log some examples
            self.logger.info("\n📝 Validation Examples:")
            for i in range(min(3, len(all_predictions))):
                self.logger.info(f"  Ref: {all_references[i]}")
                self.logger.info(f"  Pred: {all_predictions[i]}")
                self.logger.info("")
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, filename, epoch, val_loss, metrics):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics or {},
            'config': self.config.__dict__,
            'model_config': self.model.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train PictoNMT Standalone")
    
    # Data arguments
    parser.add_argument('--data_dir', default='data/processed', help='Data directory')
    parser.add_argument('--lmdb_path', default='data/cache/images/pictograms.lmdb', help='LMDB path for images')
    parser.add_argument('--metadata_file', default='data/metadata/arasaac_metadata.json', help='Metadata file')
    
    # Model arguments
    parser.add_argument('--model_config', default=None, help='Model config JSON file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--save_every', type=int, default=2, help='Save checkpoint every N epochs')
    
    # Output arguments
    parser.add_argument('--output_dir', default='models/standalone', help='Output directory')
    parser.add_argument('--experiment_name', default='pictonmt_standalone', help='Experiment name')
    
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    
    print(f"Starting PictoNMT training")
    print(f"Output directory: {args.output_dir}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
    
    # Create model config
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        model_config = create_model_config(len(tokenizer))
    
    # Create model
    print("Creating model...")
    model = PictoNMT(vocab_size=len(tokenizer), config=model_config)
    print(f"Model size: {model.get_model_size()}")
    
    # Set up data
    print("Loading datasets...")
    image_processor = ImageProcessor(lmdb_path=args.lmdb_path, resolution=224)
    
    train_dataset = PictoDataset(
        data_file=os.path.join(args.data_dir, "train"),
        metadata_file=os.path.join(args.data_dir, "train.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=100
    )
    
    val_dataset = PictoDataset(
        data_file=os.path.join(args.data_dir, "valid"),
        metadata_file=os.path.join(args.data_dir, "valid.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=100
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = StandaloneTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=args
    )
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()