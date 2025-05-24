import os
import sys
import json
import traceback
import shutil
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class StandalonePhase1Tester:
    """Comprehensive tester for standalone Phase 1 components"""
    
    def __init__(self):
        self.project_root = project_root
        self.passed_tests = []
        self.failed_tests = []
        self.warnings = []
        
    def log_success(self, test_name):
        print(f"✅ {test_name}")
        self.passed_tests.append(test_name)
    
    def log_failure(self, test_name, error):
        print(f"❌ {test_name}")
        print(f"   Error: {error}")
        self.failed_tests.append((test_name, str(error)))
    
    def log_warning(self, test_name, warning):
        print(f"⚠️  {test_name}")
        print(f"   Warning: {warning}")
        self.warnings.append((test_name, str(warning)))
    
    def print_summary(self):
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        print("\n" + "="*60)
        print("STANDALONE PHASE 1 TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {len(self.passed_tests)}/{total_tests}")
        print(f"❌ Failed: {len(self.failed_tests)}/{total_tests}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.failed_tests:
            print("\nFAILED TESTS:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error}")
        
        print("="*60)
        return len(self.failed_tests) == 0

    def test_1_environment_setup(self):
        """Test 1: Basic environment setup"""
        print("\n🔧 Testing Environment Setup...")
        
        # Test Python version
        try:
            import sys
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor != 10:
                self.log_warning("Python Version", f"Expected Python 3.10, got {python_version.major}.{python_version.minor}")
            else:
                self.log_success("Python Version (3.10)")
        except Exception as e:
            self.log_failure("Python Version", e)
        
        # Test essential packages (NO EOLE)
        essential_packages = [
            ('torch', 'PyTorch'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('PIL', 'Pillow'),
            ('requests', 'Requests'),
            ('tqdm', 'TQDM'),
            ('lmdb', 'LMDB'),
            ('matplotlib', 'Matplotlib')
        ]
        
        for package, name in essential_packages:
            try:
                __import__(package)
                self.log_success(f"{name} Import")
            except ImportError as e:
                self.log_failure(f"{name} Import", e)
        
        # Test transformers
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
            self.log_success("Transformers + FlauBERT Tokenizer")
        except Exception as e:
            self.log_failure("Transformers + FlauBERT Tokenizer", e)

    def test_2_project_structure(self):
        """Test 2: Standalone project structure"""
        print("\n📁 Testing Standalone Project Structure...")
        
        required_dirs = [
            "data",
            "scripts",
            "src/pictollms",
            "src/pictollms/data",
            "src/pictollms/models",
            "src/pictollms/models/complete",  # Focus on complete standalone models
            "src/pictollms/eval"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"Directory: {dir_path}")
            else:
                self.log_failure(f"Directory: {dir_path}", "Directory not found")
        
        # Test standalone package import
        try:
            import pictollms
            self.log_success("PictoLLMs Package Import")
        except ImportError as e:
            self.log_failure("PictoLLMs Package Import", e)

    def test_3_data_processing_pipeline(self):
        """Test 3: Complete data processing pipeline"""
        print("\n💾 Testing Data Processing Pipeline...")
        
        # Create sample PropictoOrféo data
        sample_data = [
            {
                "sentence": "Je mange une pomme",
                "pictos": [2627, 11317, 37406, 2627],
                "pictos_tokens": "je manger une pomme"
            },
            {
                "sentence": "Il va au magasin", 
                "pictos": [5621, 3254, 8732],
                "pictos_tokens": "il aller magasin"
            }
        ]
        
        # Create test directories
        test_dir = self.project_root / "data" / "test_pipeline"
        test_dir.mkdir(exist_ok=True)
        
        source_dir = test_dir / "source"
        source_dir.mkdir(exist_ok=True)
        
        # Save sample data
        sample_file = source_dir / "test_corpus.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        try:
            # Test PropictoOrféo processing
            from scripts.data_processing.process_propicto import process_propicto_files
            
            output_file = str(test_dir / "processed_data.json")
            process_propicto_files(str(source_dir), output_file)
            
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                if len(processed_data) > 0 and 'pictogram_sequence' in processed_data[0]:
                    self.log_success("PropictoOrféo Processing")
                else:
                    self.log_failure("PropictoOrféo Processing", "Invalid output format")
            else:
                self.log_failure("PropictoOrféo Processing", "No output file created")
            
            # Clean up
            shutil.rmtree(test_dir)
            
        except Exception as e:
            self.log_failure("Data Processing Pipeline", e)
            if test_dir.exists():
                shutil.rmtree(test_dir)

    def test_4_arasaac_client(self):
        """Test 4: ARASAAC client (standalone)"""
        print("\n🔗 Testing ARASAAC Client...")
        
        try:
            from pictollms.data.arasaac_client import ArasaacClient
            
            cache_dir = self.project_root / "data" / "cache" / "test_arasaac"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            client = ArasaacClient(cache_dir=str(cache_dir))
            self.log_success("ARASAAC Client Initialization")
            
            # Test metadata retrieval (optional - requires internet)
            try:
                test_picto_id = 2627
                metadata = client.get_pictogram_metadata(test_picto_id)
                
                if metadata and isinstance(metadata, dict):
                    self.log_success("ARASAAC Metadata Retrieval")
                else:
                    self.log_warning("ARASAAC Metadata Retrieval", "No metadata or wrong format")
            except Exception as e:
                self.log_warning("ARASAAC Metadata Retrieval", f"Network error: {e}")
            
            # Clean up
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                
        except Exception as e:
            self.log_failure("ARASAAC Client", e)

    def test_5_image_processor(self):
        """Test 5: Standalone image processor"""
        print("\n🖼️  Testing Image Processor...")
        
        try:
            from pictollms.data.image_processor import ImageProcessor
            
            # Test with non-existent LMDB (should handle gracefully)
            processor = ImageProcessor("nonexistent.lmdb", resolution=224)
            self.log_success("Image Processor Initialization")
            
            # Test getting empty image
            empty_img = processor.get_image(12345)
            
            if empty_img is not None and empty_img.shape == (3, 224, 224):
                self.log_success("Image Processor Empty Image Handling")
            else:
                self.log_failure("Image Processor Empty Image", f"Wrong shape: {empty_img.shape if empty_img is not None else 'None'}")
            
            # Test batch processing
            batch_imgs = processor.get_batch_images([1, 2, 3])
            if batch_imgs.shape == (3, 3, 224, 224):
                self.log_success("Image Processor Batch Processing")
            else:
                self.log_failure("Image Processor Batch", f"Wrong shape: {batch_imgs.shape}")
                
        except Exception as e:
            self.log_failure("Image Processor", e)

    def test_6_standalone_dataset(self):
        """Test 6: Standalone dataset loading"""
        print("\n📚 Testing Standalone Dataset...")
        
        try:
            from pictollms.data.dataset import PictoDataset
            from pictollms.data.image_processor import ImageProcessor
            from transformers import AutoTokenizer
            
            # Create test data files
            test_dir = self.project_root / "data" / "test_dataset"
            test_dir.mkdir(exist_ok=True)
            
            # Create .picto file
            with open(test_dir / "test.picto", 'w') as f:
                f.write("1 2 3\n4 5 6\n")
            
            # Create .fr file
            with open(test_dir / "test.fr", 'w') as f:
                f.write("première phrase\ndeuxième phrase\n")
            
            # Create metadata file
            with open(test_dir / "test.meta.json", 'w') as f:
                json.dump([
                    {"pictogram_sequence": [1, 2, 3]},
                    {"pictogram_sequence": [4, 5, 6]}
                ], f)
            
            # Initialize components
            tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
            image_processor = ImageProcessor("nonexistent.lmdb")
            
            # Create dataset
            dataset = PictoDataset(
                data_file=str(test_dir / "test"),
                metadata_file=str(test_dir / "test.meta.json"),
                image_processor=image_processor,
                tokenizer=tokenizer,
                max_length=50
            )
            
            if len(dataset) == 2:
                self.log_success("Dataset Length")
            else:
                self.log_failure("Dataset Length", f"Expected 2, got {len(dataset)}")
            
            # Test dataset item
            item = dataset[0]
            expected_keys = ['pictogram_sequence', 'target_text', 'metadata', 'images']
            
            if all(key in item for key in expected_keys):
                self.log_success("Dataset Item Structure")
            else:
                missing = [key for key in expected_keys if key not in item]
                self.log_failure("Dataset Item Structure", f"Missing keys: {missing}")
            
            # Clean up
            shutil.rmtree(test_dir)
            
        except Exception as e:
            self.log_failure("Standalone Dataset", e)

    def test_7_core_model_components(self):
        """Test 7: Core standalone model components (NO EOLE)"""
        print("\n🏗️  Testing Core Model Components...")
        
        standalone_components = [
            ('pictollms.models.encoders.visual_encoder', 'VisualEncoder'),
            ('pictollms.models.encoders.semantic_encoder', 'SemanticEncoder'),
            ('pictollms.models.encoders.dual_path_encoder', 'DualPathEncoder'),
            ('pictollms.models.decoders.transformer_decoder', 'TransformerDecoder'),
            ('pictollms.models.schema.schema_inducer', 'SchemaInducer'),
            ('pictollms.models.complete.pictonmt', 'PictoNMT'),  # Main standalone model
            ('pictollms.decoding.beam_search', 'GreedySearch'),
            ('pictollms.decoding.beam_search', 'BeamSearch'),
            ('pictollms.decoding.beam_search', 'CAsiBeamSearch')
        ]
        
        for module_name, class_name in standalone_components:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                self.log_success(f"Model Component: {class_name}")
            except Exception as e:
                self.log_failure(f"Model Component: {class_name}", e)

    def test_8_complete_pictonmt_model(self):
        """Test 8: Complete standalone PictoNMT model"""
        print("\n🎯 Testing Complete PictoNMT Model...")
        
        try:
            from pictollms.models.complete.pictonmt import PictoNMT, create_model_config
            from transformers import AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
            
            # Create model config
            config = create_model_config(len(tokenizer))
            
            # Create model
            model = PictoNMT(vocab_size=len(tokenizer), config=config)
            self.log_success("PictoNMT Model Creation")
            
            # Test model size calculation
            if hasattr(model, 'get_model_size'):
                size_info = model.get_model_size()
                self.log_success(f"Model Size Calculation: {size_info}")
            
            # Test forward pass with dummy data
            batch_size = 2
            seq_len = 5
            
            dummy_batch = {
                'images': torch.zeros(batch_size, seq_len, 3, 224, 224),
                'categories': torch.zeros(batch_size, seq_len, 5, dtype=torch.long),
                'types': torch.zeros(batch_size, seq_len, dtype=torch.long),
                'attention_mask': torch.ones(batch_size, seq_len),
                'target_ids': torch.randint(0, len(tokenizer), (batch_size, 10))
            }
            
            # Test training mode
            
            outputs = model(dummy_batch, mode='train')
            
            if 'logits' in outputs and outputs['logits'].shape[-1] == len(tokenizer):
                self.log_success("PictoNMT Forward Pass (Training)")
            else:
                self.log_failure("PictoNMT Forward Pass", "Wrong output format")
            
            # Test inference mode
            inf_outputs = model(dummy_batch, mode='inference')
            if 'encoder_outputs' in inf_outputs and 'schema' in inf_outputs:
                self.log_success("PictoNMT Forward Pass (Inference)")
            else:
                self.log_failure("PictoNMT Inference", "Missing required outputs")
                
        except Exception as e:
            self.log_failure("Complete PictoNMT Model", e)

    def test_9_evaluation_metrics(self):
        """Test 9: Evaluation metrics"""
        print("\n📊 Testing Evaluation Metrics...")
        
        try:
            from pictollms.eval.metrics import evaluate_translations
            
            # Test with sample data
            predictions = ["je mange une pomme", "il va au magasin"]
            references = ["je mange une pomme rouge", "il va au magasin local"]
            
            metrics = evaluate_translations(predictions, references)
            
            expected_metrics = ['bleu', 'rouge_l', 'content_preservation', 'functional_word_accuracy']
            
            if all(metric in metrics for metric in expected_metrics):
                self.log_success("Evaluation Metrics")
                print(f"   Sample BLEU: {metrics['bleu']:.2f}")
                print(f"   Sample ROUGE-L: {metrics['rouge_l']:.2f}")
            else:
                missing = [m for m in expected_metrics if m not in metrics]
                self.log_failure("Evaluation Metrics", f"Missing: {missing}")
                
        except Exception as e:
            self.log_failure("Evaluation Metrics", e)

    def test_10_fix_standalone_imports(self):
        """Test 10: Fix and verify standalone imports"""
        print("\n🔧 Fixing Standalone Imports...")
        
        # Create clean models __init__.py (NO EOLE)
        models_init = self.project_root / "src" / "pictollms" / "models" / "__init__.py"
        
        try:
            clean_content = '''"""
PictoNMT Models Package - Standalone Implementation
No Eole dependencies
"""

__all__ = []

# Core encoders
try:
    from pictollms.models.encoders.visual_encoder import VisualEncoder
    from pictollms.models.encoders.semantic_encoder import SemanticEncoder  
    from pictollms.models.encoders.dual_path_encoder import DualPathEncoder
    
    __all__.extend(['VisualEncoder', 'SemanticEncoder', 'DualPathEncoder'])
    print("✅ Standalone encoders loaded")
    
except ImportError as e:
    print(f"❌ Error loading encoders: {e}")

# Decoder
try:
    from pictollms.models.decoders.transformer_decoder import TransformerDecoder
    __all__.append('TransformerDecoder')
    print("✅ Transformer decoder loaded")
    
except ImportError as e:
    print(f"❌ Error loading decoder: {e}")

# Schema components
try:
    from pictollms.models.schema.schema_inducer import SchemaInducer
    __all__.append('SchemaInducer')
    print("✅ Schema inducer loaded")
    
except ImportError as e:
    print(f"❌ Error loading schema inducer: {e}")

# Complete model
try:
    from pictollms.models.complete.pictonmt import PictoNMT
    __all__.append('PictoNMT')
    print("✅ Complete PictoNMT model loaded")
    
except ImportError as e:
    print(f"❌ Error loading complete model: {e}")

print(f"📦 Available standalone models: {__all__}")
'''
            
            # Backup existing file
            if models_init.exists():
                backup_path = models_init.with_suffix('.py.eole_backup')
                with open(models_init, 'r') as f:
                    with open(backup_path, 'w') as bf:
                        bf.write(f.read())
            
            # Write clean standalone version
            with open(models_init, 'w') as f:
                f.write(clean_content)
            
            self.log_success("Fixed Standalone Models Import")
            
        except Exception as e:
            self.log_failure("Fix Standalone Imports", e)

    def run_all_tests(self):
        """Run all standalone Phase 1 tests"""
        print("🚀 Starting Standalone Phase 1 Testing Suite")
        print("Testing Environment Setup and Data Pipeline (NO EOLE)")
        print("="*60)
        
        test_methods = [
            self.test_1_environment_setup,
            self.test_2_project_structure,
            self.test_3_data_processing_pipeline,
            self.test_4_arasaac_client,
            self.test_5_image_processor,
            self.test_6_standalone_dataset,
            self.test_7_core_model_components,
            self.test_8_complete_pictonmt_model,
            self.test_9_evaluation_metrics,
            self.test_10_fix_standalone_imports
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_failure(test_name, f"Test crashed: {e}")
                traceback.print_exc()
        
        return self.print_summary()

def main():
    """Main testing function"""
    tester = StandalonePhase1Tester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 Standalone Phase 1 testing completed successfully!")
        print("✅ Ready to proceed to Phase 2: Standalone Model Training")
        print("\nNext steps:")
        print("1. Add your PropictoOrféo data to data/propicto-source/")
        print("2. Run: python scripts/training/train.py")
        print("3. Test the trained model")
    else:
        print("\n❌ Standalone Phase 1 testing completed with failures.")
        print("🔧 Please fix the issues above before proceeding.")
    
    return success

if __name__ == "__main__":
    main()