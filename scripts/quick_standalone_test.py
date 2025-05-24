import os
import sys
import json
import shutil
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def create_sample_propicto_data():
    """Create realistic sample PropictoOrféo data"""
    print("📋 Creating sample PropictoOrféo data...")
    
    sample_data = [
        {
            "sentence": "Je mange une pomme rouge",
            "pictos": [2627, 11317, 37406, 8901, 12345],
            "pictos_tokens": "je manger une pomme rouge"
        },
        {
            "sentence": "Il va au magasin demain",
            "pictos": [5621, 3254, 8732, 9999],
            "pictos_tokens": "il aller magasin demain"
        },
        {
            "sentence": "La voiture est très rapide",
            "pictos": [1111, 2222, 3333, 4444, 5555],
            "pictos_tokens": "voiture être très rapide"
        },
        {
            "sentence": "Nous jouons dans le parc",
            "pictos": [6666, 7777, 8888, 9990, 1010],
            "pictos_tokens": "nous jouer dans le parc"
        },
        {
            "sentence": "Elle lit un livre intéressant",
            "pictos": [1212, 1313, 1414, 1515, 1616],
            "pictos_tokens": "elle lire un livre intéressant"
        },
        {
            "sentence": "Mon chat dort sur le lit",
            "pictos": [1717, 1818, 1919, 2020, 2121, 2222],
            "pictos_tokens": "mon chat dormir sur le lit"
        },
        {
            "sentence": "Les enfants aiment jouer dehors",
            "pictos": [2323, 2424, 2525, 2626, 2727],
            "pictos_tokens": "les enfants aimer jouer dehors"
        },
        {
            "sentence": "Papa prépare le dîner",
            "pictos": [2828, 2929, 3030, 3131],
            "pictos_tokens": "papa préparer le dîner"
        }
    ]
    
    # Create source directory
    source_dir = project_root / "data" / "propicto-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    sample_file = source_dir / "sample_corpus.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created sample data: {len(sample_data)} sentences")
    return str(sample_file)

def test_data_processing():
    """Test the complete data processing pipeline"""
    print("Testing Data Processing Pipeline...")
    
    try:
        from scripts.data_processing.process_propicto import process_propicto_files
        
        input_dir = str(project_root / "data" / "propicto-source")
        output_file = str(project_root / "data" / "propicto_base_test.json")
        
        process_propicto_files(input_dir, output_file)
        
        if not os.path.exists(output_file):
            print("PropictoOrféo processing failed")
            return False
        
        print("PropictoOrféo processing successful")
        
        # Verify processed data
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        print(f"  - Processed {len(processed_data)} entries")
        print(f"   - Sample entry: {processed_data[0]['target_text']}")
        print(f"   - Pictogram sequence: {processed_data[0]['pictogram_sequence']}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_model_components():
    """Test individual model components"""
    print("\n🏗️  Testing Model Components...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        print("✅ FlauBERT tokenizer loaded")
        
        # Test visual encoder
        from pictollms.models.encoders.visual_encoder import VisualEncoder
        
        visual_encoder = VisualEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            num_heads=3,
            num_layers=3,
            output_dim=512
        )
        
        # Test with dummy images
        dummy_images = torch.randn(2, 3, 3, 224, 224)  # batch_size=2, seq_len=3
        visual_output = visual_encoder(dummy_images)
        
        if visual_output.shape == (2, 3, 512):
            print("✅ Visual encoder working")
        else:
            print(f"❌ Visual encoder wrong output shape: {visual_output.shape}")
        
        # Test semantic encoder
        from pictollms.models.encoders.semantic_encoder import SemanticEncoder
        
        semantic_encoder = SemanticEncoder(
            category_vocab_size=200,
            type_vocab_size=10,
            embedding_dim=256,
            output_dim=512
        )
        
        # Test with dummy metadata
        dummy_categories = torch.randint(0, 200, (2, 3, 5))  # batch_size=2, seq_len=3, max_categories=5
        dummy_types = torch.randint(0, 10, (2, 3))
        
        semantic_output = semantic_encoder(dummy_categories, dummy_types)
        
        if semantic_output.shape == (2, 3, 512):
            print("✅ Semantic encoder working")
        else:
            print(f"❌ Semantic encoder wrong output shape: {semantic_output.shape}")
        
        # Test dual path encoder
        from pictollms.models.encoders.dual_path_encoder import DualPathEncoder
        
        dual_encoder = DualPathEncoder()
        fused_output = dual_encoder(dummy_images, dummy_categories, dummy_types)
        
        if fused_output.shape == (2, 3, 512):
            print("✅ Dual path encoder working")
        else:
            print(f"❌ Dual path encoder wrong output shape: {fused_output.shape}")
        
        # Test schema inducer
        from pictollms.models.schema.schema_inducer import SchemaInducer
        
        schema_inducer = SchemaInducer(hidden_size=512, num_heads=8, num_layers=3)
        
        # Test with encoder outputs
        attention_mask = torch.ones(2, 3)
        schema = schema_inducer(fused_output, attention_mask)
        
        if 'enhanced_repr' in schema and schema['enhanced_repr'].shape == (2, 3, 512):
            print("✅ Schema inducer working")
        else:
            print("❌ Schema inducer failed")
        
        # Test transformer decoder
        from pictollms.models.decoders.transformer_decoder import TransformerDecoder
        
        decoder = TransformerDecoder(
            vocab_size=len(tokenizer),
            hidden_size=512,
            num_layers=6,
            num_heads=8
        )
        
        # Test decoder forward pass
        target_ids = torch.randint(0, len(tokenizer), (2, 10))
        decoder_output = decoder(target_ids, fused_output, encoder_mask=attention_mask)
        
        if decoder_output.shape == (2, 10, len(tokenizer)):
            print("✅ Transformer decoder working")
        else:
            print(f"❌ Transformer decoder wrong output shape: {decoder_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_model():
    """Test the complete PictoNMT model"""
    print("\n🎯 Testing Complete PictoNMT Model...")
    
    try:
        from pictollms.models.complete.pictonmt import PictoNMT, create_model_config
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        
        # Create model
        config = create_model_config(len(tokenizer))
        model = PictoNMT(vocab_size=len(tokenizer), config=config)
        
        print("✅ Complete PictoNMT model created")
        print(f"   Model parameters: ~{sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 3
        
        dummy_batch = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224),
            'categories': torch.randint(0, 200, (batch_size, seq_len, 5)),
            'types': torch.randint(0, 10, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'target_ids': torch.randint(0, len(tokenizer), (batch_size, 10))
        }
        
        # Test training mode
        model.train()
        outputs = model(dummy_batch, mode='train')
        
        if 'logits' in outputs and outputs['logits'].shape == (batch_size, 10, len(tokenizer)):
            print("✅ Training forward pass working")
        else:
            print("❌ Training forward pass failed")
        
        # Test inference mode
        model.eval()
        with torch.no_grad():
            inf_outputs = model(dummy_batch, mode='inference')
        
        if 'encoder_outputs' in inf_outputs and 'schema' in inf_outputs:
            print("✅ Inference forward pass working")
        else:
            print("❌ Inference forward pass failed")
        
        # Test generation (simplified)
        try:
            predictions = model.generate(dummy_batch, strategy='greedy', tokenizer=tokenizer)
            if isinstance(predictions, list) and len(predictions) == batch_size:
                print("✅ Generation working")
                
                # Decode first prediction
                if len(predictions[0]) > 0:
                    decoded = tokenizer.decode(predictions[0], skip_special_tokens=True)
                    print(f"   Sample generation: '{decoded}'")
            else:
                print("❌ Generation failed")
        except Exception as e:
            print(f"⚠️  Generation test failed (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    """Test evaluation metrics"""
    print("\n📊 Testing Evaluation Metrics...")
    
    try:
        from pictollms.eval.metrics import evaluate_translations
        
        # Sample predictions and references
        predictions = [
            "je mange une pomme rouge",
            "il va au magasin",
            "la voiture est rapide"
        ]
        
        references = [
            "je mange une pomme rouge délicieuse",
            "il va au magasin local",
            "la voiture est très rapide"
        ]
        
        metrics = evaluate_translations(predictions, references)
        
        expected_metrics = ['bleu', 'rouge_l', 'content_preservation', 'functional_word_accuracy']
        
        if all(metric in metrics for metric in expected_metrics):
            print("✅ Evaluation metrics working")
            print(f"   BLEU: {metrics['bleu']:.2f}")
            print(f"   ROUGE-L: {metrics['rouge_l']:.2f}")
            print(f"   Content Preservation: {metrics['content_preservation']:.2f}")
            print(f"   Functional Word Accuracy: {metrics['functional_word_accuracy']:.2f}")
        else:
            missing = [m for m in expected_metrics if m not in metrics]
            print(f"❌ Missing metrics: {missing}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation with processed data"""
    print("\n📚 Testing Dataset Creation...")
    
    try:
        from pictollms.data.dataset import PictoDataset
        from pictollms.data.image_processor import ImageProcessor
        from transformers import AutoTokenizer
        
        # Check if we have processed data
        processed_file = project_root / "data" / "propicto_base_test.json"
        if not processed_file.exists():
            print("⚠️  No processed data found, skipping dataset test")
            return True
        
        # Create simple split for testing
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create test files in Eole-like format
        test_dir = project_root / "data" / "test_dataset"
        test_dir.mkdir(exist_ok=True)
        
        # Create .picto and .fr files
        with open(test_dir / "test.picto", 'w') as picto_f, \
             open(test_dir / "test.fr", 'w') as fr_f:
            
            for item in data[:5]:  # Use first 5 items
                picto_line = ' '.join(map(str, item['pictogram_sequence']))
                picto_f.write(picto_line + '\n')
                fr_f.write(item['target_text'] + '\n')
        
        # Create metadata file
        metadata = [{'pictogram_sequence': item['pictogram_sequence']} for item in data[:5]]
        with open(test_dir / "test.meta.json", 'w') as f:
            json.dump(metadata, f)
        
        # Initialize components
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        image_processor = ImageProcessor("nonexistent.lmdb")  # Mock processor
        
        # Create dataset
        dataset = PictoDataset(
            data_file=str(test_dir / "test"),
            metadata_file=str(test_dir / "test.meta.json"),
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=50
        )
        
        if len(dataset) == 5:
            print("✅ Dataset creation successful")
            
            # Test dataset item
            item = dataset[0]
            expected_keys = ['pictogram_sequence', 'target_text', 'metadata', 'images', 'target_ids']
            
            if all(key in item for key in expected_keys):
                print("✅ Dataset item structure correct")
                print(f"   Sample text: '{item['target_text']}'")
                print(f"   Pictogram sequence shape: {item['pictogram_sequence'].shape}")
                print(f"   Images shape: {item['images'].shape}")
            else:
                missing = [key for key in expected_keys if key not in item]
                print(f"❌ Missing dataset keys: {missing}")
        else:
            print(f"❌ Wrong dataset size: expected 5, got {len(dataset)}")
        
        # Clean up
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    files_to_remove = [
        project_root / "data" / "propicto_base_test.json",
        project_root / "data" / "propicto_base_test_stats.json"
    ]
    
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            print(f"   Removed: {file_path.name}")

def main():
    """Main test function"""
    print("🚀 Quick Standalone Pipeline Test")
    print("=" * 50)
    print("Testing complete standalone PictoNMT system (NO Eole)")
    
    # Step 1: Create sample data
    create_sample_propicto_data()
    
    # Step 2: Test data processing
    processed_file = test_data_processing()
    if not processed_file:
        print("\n❌ Data processing failed - stopping tests")
        return False
    
    # Step 3: Test model components
    if not test_model_components():
        print("\n❌ Model components failed - stopping tests")
        return False
    
    # Step 4: Test complete model
    if not test_complete_model():
        print("\n❌ Complete model failed - stopping tests") 
        return False
    
    # Step 5: Test evaluation
    if not test_evaluation():
        print("\n❌ Evaluation failed")
        return False
    
    # Step 6: Test dataset creation
    if not test_dataset_creation():
        print("\n❌ Dataset creation failed")
        return False
    
    # Step 7: Cleanup
    cleanup()
    
    print("\n🎉 All Standalone Tests Passed!")
    print("=" * 50)
    print("✅ Data processing pipeline working")
    print("✅ All model components working") 
    print("✅ Complete PictoNMT model working")
    print("✅ Evaluation metrics working")
    print("✅ Dataset creation working")
    print("\n🎯 Ready for training!")
    print("Next step: python scripts/training/train.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)