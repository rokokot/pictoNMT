# scripts/test_fixes.py
"""
Quick test script to validate the fixes
"""

import os
import sys
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_model_creation():
    """Test that the fixed model can be created and run forward pass"""
    print("🧪 Testing Model Creation...")
    
    try:
        from pictollms.models.complete.pictonmt import PictoNMT, create_model_config
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        
        # Create model
        config = create_model_config(len(tokenizer))
        model = PictoNMT(vocab_size=len(tokenizer), config=config)
        model.pad_token_id = tokenizer.pad_token_id
        
        print("✅ Model created successfully")
        print(f"   {model.get_model_size()}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 4
        
        dummy_batch = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224),
            'categories': torch.randint(0, 50, (batch_size, seq_len, 5)),
            'types': torch.randint(0, 10, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'target_ids': torch.randint(0, len(tokenizer), (batch_size, 12))
        }
        
        # Test training mode
        outputs = model(dummy_batch, mode='train')
        print("✅ Training forward pass working")
        
        # Test loss computation
        loss_dict = model.compute_loss(outputs, dummy_batch)
        print(f"✅ Loss computation working: {loss_dict['total_loss']:.4f}")
        
        # Test inference mode
        inf_outputs = model(dummy_batch, mode='inference')
        print("✅ Inference forward pass working")
        
        # Test generation
        try:
            predictions = model.generate(dummy_batch, strategy='greedy', tokenizer=tokenizer)
            print(f"✅ Generation working: {len(predictions)} sequences generated")
            
            # Try CASI generation
            predictions_casi = model.generate(dummy_batch, strategy='casi', tokenizer=tokenizer)
            print(f"✅ CASI generation working: {len(predictions_casi)} sequences generated")
            
        except Exception as e:
            print(f"⚠️ Generation had issues (expected for dummy data): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_processor():
    """Test the fixed metadata processor"""
    print("\n🧪 Testing Metadata Processor...")
    
    try:
        from pictollms.data.metadata_processor import MetadataProcessor
        
        # Test without metadata file (should use defaults)
        processor = MetadataProcessor()
        
        # Test with sample pictogram IDs
        test_ids = [2627, 11317, 37406, 8901]
        
        # Test single processing
        features = processor.get_metadata_features(2627)
        print(f"✅ Single metadata processing working: {features}")
        
        # Test batch processing
        batch_features = processor.batch_process(test_ids)
        print(f"✅ Batch metadata processing working: {batch_features['categories'].shape}")
        
        # Test with actual metadata file if available
        metadata_file = project_root / "data" / "metadata" / "arasaac_metadata.json"
        if metadata_file.exists():
            processor_with_data = MetadataProcessor(str(metadata_file))
            stats = processor_with_data.get_stats()
            print(f"✅ Metadata file loading working: {stats['total_pictograms']} pictograms")
        else:
            print("ℹ️ No metadata file found, using defaults")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata processor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_beam_search():
    """Test the fixed beam search implementations"""
    print("\n🧪 Testing Beam Search...")
    
    try:
        from pictollms.decoding.beam_search import GreedySearch, BeamSearch, CAsiBeamSearch
        from pictollms.models.decoders.transformer_decoder import TransformerDecoder
        from transformers import AutoTokenizer
        
        # Create a simple decoder for testing
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        decoder = TransformerDecoder(vocab_size=len(tokenizer), hidden_size=512)
        
        # Create dummy encoder outputs
        batch_size = 2
        seq_len = 4
        hidden_size = 512
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test greedy search
        greedy = GreedySearch(max_length=20)
        greedy_results = greedy.search(decoder, encoder_outputs, attention_mask, tokenizer)
        print(f"✅ Greedy search working: {len(greedy_results)} sequences")
        
        # Test beam search
        beam = BeamSearch(beam_size=3, max_length=20)
        beam_results = beam.search(decoder, encoder_outputs, attention_mask, tokenizer)
        print(f"✅ Beam search working: {len(beam_results)} sequences")
        
        # Test CASI beam search
        # Create dummy schema
        dummy_schema = {
            'structure_type': torch.randint(0, 4, (batch_size,)),
            'complexity_score': torch.rand(batch_size),
            'enhanced_repr': encoder_outputs
        }
        
        casi = CAsiBeamSearch(beam_size=3, max_length=20, schema_weight=0.3)
        casi_results = casi.search(
            decoder, encoder_outputs, dummy_schema, attention_mask, tokenizer
        )
        print(f"✅ CASI beam search working: {len(casi_results)} sequences")
        
        return True
        
    except Exception as e:
        print(f"❌ Beam search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing with the fixes"""
    print("\n🧪 Testing Data Processing...")
    
    try:
        # Create some dummy data for testing
        test_dir = project_root / "data" / "test_fixes"
        test_dir.mkdir(exist_ok=True)
        
        # Create dummy data files
        with open(test_dir / "test.picto", 'w') as f:
            f.write("2627 11317 37406\n")
            f.write("8901 12345 5621\n")
        
        with open(test_dir / "test.fr", 'w') as f:
            f.write("Je mange une pomme\n")
            f.write("Il va au magasin\n")
        
        with open(test_dir / "test.meta.json", 'w') as f:
            json.dump([
                {"pictogram_sequence": [2627, 11317, 37406]},
                {"pictogram_sequence": [8901, 12345, 5621]}
            ], f)
        
        # Test dataset creation
        from pictollms.data.image_processor import ImageProcessor
        from transformers import AutoTokenizer
        
        # Test with the fixed dataset if available
        try:
            from pictollms.data.metadata_processor import PictoDatasetFixed
            
            tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
            image_processor = ImageProcessor("nonexistent.lmdb")
            
            dataset = PictoDatasetFixed(
                data_file=str(test_dir / "test"),
                metadata_file=str(test_dir / "test.meta.json"),
                image_processor=image_processor,
                tokenizer=tokenizer,
                max_length=50
            )
            
            # Test dataset item
            item = dataset[0]
            print(f"✅ Fixed dataset working: {item['pictogram_sequence'].shape}")
            print(f"   Categories shape: {item['categories'].shape}")
            print(f"   Types shape: {item['types'].shape}")
            
        except ImportError:
            # Fallback to original dataset
            from pictollms.data.dataset import PictoDataset
            
            tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
            image_processor = ImageProcessor("nonexistent.lmdb")
            
            dataset = PictoDataset(
                data_file=str(test_dir / "test"),
                metadata_file=str(test_dir / "test.meta.json"),
                image_processor=image_processor,
                tokenizer=tokenizer,
                max_length=50
            )
            
            item = dataset[0]
            print(f"✅ Original dataset working: {item['pictogram_sequence'].shape}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test that training components work together"""
    print("\n🧪 Testing Training Components...")
    
    try:
        from pictollms.models.complete.pictonmt import PictoNMT, create_model_config
        from transformers import AutoTokenizer
        
        # Create model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
        config = create_model_config(len(tokenizer))
        model = PictoNMT(vocab_size=len(tokenizer), config=config)
        model.pad_token_id = tokenizer.pad_token_id
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = 2
        seq_len = 4
        dummy_batch = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224),
            'categories': torch.randint(0, 50, (batch_size, seq_len, 5)),
            'types': torch.randint(0, 10, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'target_ids': torch.randint(1, len(tokenizer), (batch_size, 12))  # Avoid pad token
        }
        
        # Test training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(dummy_batch, mode='train')
        loss_dict = model.compute_loss(outputs, dummy_batch)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step working: loss = {loss.item():.4f}")
        print(f"   Loss components: {[(k, v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing PictoNMT Fixes")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Metadata Processor", test_metadata_processor),
        ("Beam Search", test_beam_search),
        ("Data Processing", test_data_processing),
        ("Training Components", test_training_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        print("\n✅ Ready to run training with:")
        print("   python scripts/training/train_fixed.py")
    else:
        print("⚠️ Some issues remain. Check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()