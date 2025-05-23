import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

from pictollms.eval.metrics import evaluate_translations, calculate_schema_alignment_score

logger = logging.getLogger(__name__)

class PictoTrainer:
    
    def __init__(self, 
                 encoder, 
                 schema_inducer, 
                 beam_search, 
                 tokenizer, 
                 train_dataset, 
                 val_dataset, 
                 config):
        """
        Initialize PictoNMT trainer
        
        Args:
            encoder: PictoEoleEncoder for encoding pictogram sequences
            schema_inducer: SchemaInducer for generating linguistic schemas
            beam_search: CAsiBeamSearch for schema-guided decoding
            tokenizer: Tokenizer for French text
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration object
        """
        
        self.encoder = encoder
        self.schema_inducer = schema_inducer
        self.beam_search = beam_search
        self.tokenizer = tokenizer
        self.config = config
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Move models to device
        self.encoder.to(self.device)
        self.schema_inducer.to(self.device)
        
        # Set up data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=getattr(config, 'batch_size', 16), 
            shuffle=True,
            num_workers=getattr(config, 'num_workers', 4),
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=getattr(config, 'batch_size', 16),
            shuffle=False,
            num_workers=getattr(config, 'num_workers', 4),
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Set up optimizer
        all_parameters = list(self.encoder.parameters()) + list(self.schema_inducer.parameters())
        self.optimizer = torch.optim.AdamW(
            all_parameters,
            lr=getattr(config, 'learning_rate', 5e-5),
            weight_decay=getattr(config, 'weight_decay', 0.01),
            eps=getattr(config, 'adam_eps', 1e-8)
        )
        
        # Set up learning rate scheduler
        total_steps = len(self.train_loader) * getattr(config, 'num_epochs', 10)
        warmup_steps = int(total_steps * getattr(config, 'warmup_ratio', 0.1))
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Set up loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.schema_criterion = nn.CrossEntropyLoss()
        
        # Create output directory
        output_dir = getattr(config, 'output_dir', 'models/complete')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    def _collate_fn(self, batch):
        # Extract components
        pictogram_sequences = [item['pictogram_sequence'] for item in batch]
        images = [item['images'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        
        # Calculate padding lengths
        max_picto_len = max(len(seq) for seq in pictogram_sequences)
        max_target_len = max(len(tgt) for tgt in target_ids)
        
        # Pad pictogram sequences and create attention masks
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
                # Pad with zeros
                padding = torch.zeros(max_picto_len - img.shape[0], *img.shape[1:])
                padded_img = torch.cat([img, padding], dim=0)
            else:
                # Truncate if too long
                padded_img = img[:max_picto_len]
            padded_images.append(padded_img)
        
        padded_images = torch.stack(padded_images)
        
        # Pad target sequences
        padded_targets = []
        target_attention_masks = []
        
        for tgt in target_ids:
            padded_tgt = tgt.tolist() + [self.tokenizer.pad_token_id] * (max_target_len - len(tgt))
            tgt_mask = [1] * len(tgt) + [0] * (max_target_len - len(tgt))
            padded_targets.append(padded_tgt)
            target_attention_masks.append(tgt_mask)
        
        # Create metadata tensors (simplified for now)
        batch_size = len(batch)
        categories = torch.zeros(batch_size, max_picto_len, 5, dtype=torch.long)
        types = torch.zeros(batch_size, max_picto_len, dtype=torch.long)
        
        return {
            'pictogram_sequences': torch.tensor(padded_picto_seqs),
            'images': padded_images,
            'categories': categories,
            'types': types,
            'attention_masks': torch.tensor(attention_masks),
            'target_ids': torch.tensor(padded_targets),
            'target_attention_masks': torch.tensor(target_attention_masks),
            'target_texts': target_texts
        }
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {getattr(self.config, 'num_epochs', 10)} epochs")
        logger.info(f"Training dataset size: {len(self.train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_loader.dataset)}")
        logger.info(f"Batch size: {getattr(self.config, 'batch_size', 16)}")
        logger.info(f"Learning rate: {getattr(self.config, 'learning_rate', 5e-5)}")
        
        num_epochs = getattr(self.config, 'num_epochs', 10)
        eval_steps = getattr(self.config, 'eval_steps', len(self.train_loader) // 2)
        save_steps = getattr(self.config, 'save_steps', len(self.train_loader))
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            epoch_train_loss = self._train_epoch(epoch, eval_steps, save_steps)
            
            # Validation phase
            epoch_val_loss, val_metrics = self._validate_epoch(epoch)
            
            # Update tracking
            self.train_losses.append(epoch_train_loss)
            self.val_losses.append(epoch_val_loss)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1} Results:")
            logger.info(f"  Train Loss: {epoch_train_loss:.4f}")
            logger.info(f"  Val Loss: {epoch_val_loss:.4f}")
            if val_metrics:
                logger.info(f"  Val BLEU: {val_metrics.get('bleu', 0):.2f}")
                logger.info(f"  Val ROUGE-L: {val_metrics.get('rouge_l', 0):.2f}")
                logger.info(f"  Content Preservation: {val_metrics.get('content_preservation', 0):.2f}")
            
            # Save best model
            is_best = epoch_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = epoch_val_loss
                self.best_metrics = val_metrics or {}
                logger.info(f"New best model: Val Loss: {epoch_val_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, epoch_val_loss, val_metrics, is_best)
            
            # Early stopping check
            if hasattr(self.config, 'early_stopping_patience'):
                # Simple early stopping implementation
                if epoch > getattr(self.config, 'early_stopping_patience', 5):
                    recent_losses = self.val_losses[-getattr(self.config, 'early_stopping_patience', 5):]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best metrics: {self.best_metrics}")
        
        # Save final training stats
        self._save_training_stats()
    
    def _train_epoch(self, epoch, eval_steps, save_steps):
        """Train for one epoch"""
        self.encoder.train()
        self.schema_inducer.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss, loss_components = self._forward_step(batch, training=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.schema_inducer.parameters()), 
                getattr(self.config, 'max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update tracking
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/num_batches:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Periodic evaluation
            if self.global_step % eval_steps == 0:
                logger.info(f"\nStep {self.global_step} - Running validation...")
                val_loss, val_metrics = self._validate_epoch(epoch, full_eval=False)
                logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}")
                
                # Save if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_metrics = val_metrics or {}
                    self._save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                
                # Back to training mode
                self.encoder.train()
                self.schema_inducer.train()
        
        return epoch_loss / num_batches
    
    def _forward_step(self, batch, training=True):
        """Forward step through the complete model"""
        # Prepare encoder inputs
        encoder_inputs = {
            'images': batch['images'],
            'categories': batch['categories'],
            'types': batch['types']
        }
        
        # Encode pictograms
        encoder_final, memory_bank, lengths = self.encoder(encoder_inputs)
        
        # Generate schema
        schema = self.schema_inducer(memory_bank, batch['attention_masks'])
        
        # For training, we need to compute loss
        # This is a simplified implementation - in practice you'd implement proper decoder training
        if training:
            # Simplified loss computation
            # In a complete implementation, you would:
            # 1. Use teacher forcing with the decoder
            # 2. Compute cross-entropy loss against target tokens
            # 3. Add schema-based auxiliary losses
            
            batch_size, seq_len = batch['target_ids'].shape
            vocab_size = len(self.tokenizer)
            
            # Simulate decoder logits
            #  use a simple projection from the enhanced representations
            enhanced_repr = schema['enhanced_repr']  # [batch_size, seq_len, hidden_size]
            
            # Simple projection to vocabulary
            if not hasattr(self, 'vocab_projection'):
                hidden_size = enhanced_repr.shape[-1]
                self.vocab_projection = nn.Linear(hidden_size, vocab_size).to(self.device)
            
            # Project to vocabulary space
            logits = self.vocab_projection(enhanced_repr)  # [batch_size, seq_len, vocab_size]
            
            # Compute main loss
            main_loss = self.criterion(
                logits.view(-1, vocab_size), 
                batch['target_ids'].view(-1)
            )
            
            # Compute schema auxiliary losses
            schema_losses = {}
            
            # Structure type loss (if we had ground truth structure labels)
            if 'structure_logits' in schema:
                pass
            
            # Combine losses
            total_loss = main_loss
            
            loss_components = {
                'main_loss': main_loss.item(),
                'total_loss': total_loss.item()
            }
            
            return total_loss, loss_components
        
        else:
            # For evaluation, return the schema for decoding
            return schema
    
    def _validate_epoch(self, epoch, full_eval=True):
        """Validate for one epoch"""
        self.encoder.eval()
        self.schema_inducer.eval()
        
        val_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        all_schemas = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if full_eval:
                    # Generate predictions using CASI beam search
                    schema = self._forward_step(batch, training=False)
                    
                    # Prepare inputs for beam search
                    encoder_inputs = {
                        'images': batch['images'],
                        'categories': batch['categories'],
                        'types': batch['types']
                    }
                    encoder_final, memory_bank, lengths = self.encoder(encoder_inputs)
                    
                    # Generate translations
                    predictions = self.beam_search.search(
                        model=self._create_mock_model(),  # We need a decoder model here
                        encoder_outputs=memory_bank,
                        schema=schema,
                        tokenizer=self.tokenizer,
                        attention_mask=batch['attention_masks']
                    )
                    
                    # Decode predictions
                    decoded_predictions = [
                        self.tokenizer.decode(pred, skip_special_tokens=True) 
                        for pred in predictions
                    ]
                    
                    all_predictions.extend(decoded_predictions)
                    all_references.extend(batch['target_texts'])
                    all_schemas.extend([schema] * len(decoded_predictions))
                
                # Compute validation loss
                loss, _ = self._forward_step(batch, training=True)
                val_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'avg_val_loss': f"{val_loss/num_batches:.4f}"
                })
                
                # For quick validation, don't process all batches
                if not full_eval and batch_idx >= 10:
                    break
        
        avg_val_loss = val_loss / num_batches
        
        # Compute metrics if we have predictions
        metrics = None
        if full_eval and all_predictions:
            metrics = evaluate_translations(all_predictions, all_references)
            
            # Add schema alignment score
            if all_schemas:
                schema_alignment = calculate_schema_alignment_score(all_predictions, all_schemas)
                metrics['schema_alignment'] = schema_alignment
            
            # Log some examples
            logger.info("\nValidation Examples:")
            for i in range(min(3, len(all_predictions))):
                logger.info(f"  Reference: {all_references[i]}")
                logger.info(f"  Prediction: {all_predictions[i]}")
                logger.info("")
        
        return avg_val_loss, metrics
    

    #mock
    def _create_mock_model(self):
        """mock model for beam search (temporary demo solution)"""
        class MockModel:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
            
            class MockDecoder:
                def __init__(self, vocab_size):
                    self.vocab_size = vocab_size
                
                def __call__(self, input_ids, encoder_hidden_states, encoder_attention_mask=None):
                    batch_size, seq_len = input_ids.shape
                    logits = torch.randn(batch_size, seq_len, self.vocab_size, device=input_ids.device)
                    
                    class MockOutput:
                        def __init__(self, logits):
                            self.logits = logits
                    
                    return MockOutput(logits)
            
            def __init__(self, vocab_size):
                self.decoder = self.MockDecoder(vocab_size)
        
        return MockModel(len(self.tokenizer))
    
    def _move_batch_to_device(self, batch):
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _save_checkpoint(self, epoch, val_loss, metrics=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'schema_inducer_state_dict': self.schema_inducer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics or {},
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        # Keep only last N checkpoints to save space
        max_checkpoints = getattr(self.config, 'max_checkpoints', 3)
        self._cleanup_old_checkpoints(max_checkpoints)
    
    def _cleanup_old_checkpoints(self, max_checkpoints):
        """Remove old checkpoints to save disk space"""
        checkpoint_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                epoch_num = int(file.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch_num, file))
        
        # Sort by epoch and keep only the most recent ones
        checkpoint_files.sort(key=lambda x: x[0])
        
        if len(checkpoint_files) > max_checkpoints:
            files_to_remove = checkpoint_files[:-max_checkpoints]
            for epoch_num, filename in files_to_remove:
                file_path = os.path.join(self.output_dir, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old checkpoint: {filename}")
                except OSError:
                    pass
    
    def _save_training_stats(self):
        """Save training statistics"""
        stats = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'total_steps': self.global_step,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else str(self.config)
        }
        
        stats_path = os.path.join(self.output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"training statistics saved to {stats_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.schema_inducer.load_state_dict(checkpoint['schema_inducer_state_dict'])
        
        # Load optimizer and scheduler
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.best_metrics = checkpoint.get('metrics', {})
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f" Checkpoint loaded successfully")
        logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"   Global step: {self.global_step}")
        logger.info(f"   Best val loss: {self.best_val_loss:.4f}")
    
    def evaluate(self, test_dataset):
        """Evaluate model on test dataset"""
        logger.info("🔍 Running evaluation on test dataset")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=getattr(self.config, 'batch_size', 16),
            shuffle=False,
            num_workers=getattr(self.config, 'num_workers', 4),
            collate_fn=self._collate_fn
        )
        
        self.encoder.eval()
        self.schema_inducer.eval()
        
        all_predictions = []
        all_references = []
        all_schemas = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)
                
                # Generate schema
                schema = self._forward_step(batch, training=False)
                
                # Prepare inputs for beam search
                encoder_inputs = {
                    'images': batch['images'],
                    'categories': batch['categories'],
                    'types': batch['types']
                }
                encoder_final, memory_bank, lengths = self.encoder(encoder_inputs)
                
                # Generate translations
                predictions = self.beam_search.search(
                    model=self._create_mock_model(),
                    encoder_outputs=memory_bank,
                    schema=schema,
                    tokenizer=self.tokenizer,
                    attention_mask=batch['attention_masks']
                )
                
                # Decode predictions
                decoded_predictions = [
                    self.tokenizer.decode(pred, skip_special_tokens=True) 
                    for pred in predictions
                ]
                
                all_predictions.extend(decoded_predictions)
                all_references.extend(batch['target_texts'])
                all_schemas.extend([schema] * len(decoded_predictions))
        
        # Compute comprehensive metrics
        metrics = evaluate_translations(all_predictions, all_references)
        
        # Add schema alignment score
        schema_alignment = calculate_schema_alignment_score(all_predictions, all_schemas)
        metrics['schema_alignment'] = schema_alignment
        
        # Log results
        logger.info("Test Results:")
        logger.info(f"  BLEU Score: {metrics['bleu']:.2f}")
        logger.info(f"  ROUGE-L Score: {metrics['rouge_l']:.2f}")
        logger.info(f"  Content Preservation: {metrics['content_preservation']:.2f}")
        logger.info(f"  Functional Word Accuracy: {metrics['functional_word_accuracy']:.2f}")
        logger.info(f"  Schema Alignment: {metrics['schema_alignment']:.2f}")
        
        # Save detailed results
        results = {
            'metrics': metrics,
            'predictions': all_predictions[:100],  # Save first 100 examples
            'references': all_references[:100],
            'num_examples': len(all_predictions)
        }
        
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {results_path}")
        
        return metrics