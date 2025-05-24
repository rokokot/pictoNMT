# scripts/evaluation/evaluate_model.py
import os
import sys
import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pictollms.models.complete.pictonmt import PictoNMT
from pictollms.data.dataset import PictoDataset
from pictollms.data.image_processor import ImageProcessor
from pictollms.eval.metrics import evaluate_translations

def collate_fn(batch, tokenizer):
    """Collate function for evaluation"""
    # Extract components
    pictogram_sequences = [item['pictogram_sequence'] for item in batch]
    images = [item['images'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    # Pad sequences
    max_picto_len = max(len(seq) for seq in pictogram_sequences)
    
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
    
    # Create metadata (simplified)
    batch_size = len(batch)
    categories = torch.zeros(batch_size, max_picto_len, 5, dtype=torch.long)
    types = torch.zeros(batch_size, max_picto_len, dtype=torch.long)
    
    return {
        'images': torch.stack(padded_images),
        'categories': categories,
        'types': types,
        'attention_mask': torch.tensor(attention_masks),
        'target_texts': target_texts
    }

def evaluate_model(model_path, data_dir, output_dir, strategies=['greedy', 'beam', 'casi']):
    """Evaluate model with different decoding strategies"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('config', {})
    
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
    
    model = PictoNMT(vocab_size=len(tokenizer), config=model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Load test dataset
    print("Loading test dataset...")
    image_processor = ImageProcessor(
        lmdb_path="data/cache/images/pictograms.lmdb",
        resolution=224
    )
    
    test_dataset = PictoDataset(
        data_file=os.path.join(data_dir, "test"),
        metadata_file=os.path.join(data_dir, "test.meta.json"),
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=100
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,  # Smaller batch size for evaluation
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluate each strategy
    results = {}
    
    for strategy in strategies:
        print(f"\nEvaluating {strategy} decoding...")
        
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"{strategy} decoding"):
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Generate translations
                    predictions = model.generate(batch, strategy=strategy, tokenizer=tokenizer)
                    
                    # Decode predictions
                    decoded_preds = [
                        tokenizer.decode(pred, skip_special_tokens=True) 
                        for pred in predictions
                    ]
                    
                    all_predictions.extend(decoded_preds)
                    all_references.extend(batch['target_texts'])
                    
                except Exception as e:
                    print(f"Error in batch generation: {e}")
                    # Add empty predictions to maintain alignment
                    all_predictions.extend([''] * len(batch['target_texts']))
                    all_references.extend(batch['target_texts'])
        
        # Calculate metrics
        try:
            metrics = evaluate_translations(all_predictions, all_references)
            metrics['total_samples'] = len(all_predictions)
        except Exception as e:
            print(f"Error calculating metrics for {strategy}: {e}")
            metrics = {'bleu': 0, 'rouge_l': 0, 'content_preservation': 0, 'total_samples': len(all_predictions)}
        
        results[strategy] = {
            'metrics': metrics,
            'predictions': all_predictions[:20],  # Save first 20 for inspection
            'references': all_references[:20]
        }
        
        print(f"{strategy} Results:")
        print(f"  BLEU: {metrics['bleu']:.2f}")
        print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")
        print(f"  Content Preservation: {metrics['content_preservation']:.2f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        'model_path': model_path,
        'test_samples': len(test_dataset),
        'strategy_comparison': {
            strategy: {
                'bleu': results[strategy]['metrics']['bleu'],
                'rouge_l': results[strategy]['metrics']['rouge_l'],
                'content_preservation': results[strategy]['metrics']['content_preservation']
            }
            for strategy in strategies
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save examples
    with open(os.path.join(output_dir, 'translation_examples.txt'), 'w', encoding='utf-8') as f:
        f.write("PictoNMT Translation Examples\n")
        f.write("=" * 50 + "\n\n")
        
        for i in range(min(10, len(results['greedy']['predictions']))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Reference: {results['greedy']['references'][i]}\n")
            
            for strategy in strategies:
                if i < len(results[strategy]['predictions']):
                    f.write(f"{strategy.capitalize()}: {results[strategy]['predictions'][i]}\n")
            
            f.write("\n")
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    
    # Print comparison
    print(f"\nStrategy Comparison:")
    print(f"{'Strategy':<10} {'BLEU':<8} {'ROUGE-L':<8} {'Content':<8}")
    print("-" * 40)
    
    for strategy in strategies:
        metrics = results[strategy]['metrics']
        print(f"{strategy.capitalize():<10} "
              f"{metrics['bleu']:<8.2f} "
              f"{metrics['rouge_l']:<8.2f} "
              f"{metrics['content_preservation']:<8.2f}")
    
    return results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate PictoNMT model")
    
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_dir', default='data/processed', help='Data directory')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--strategies', nargs='+', default=['greedy', 'beam', 'casi'],
                       help='Decoding strategies to evaluate')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        strategies=args.strategies
    )

if __name__ == "__main__":
    main()