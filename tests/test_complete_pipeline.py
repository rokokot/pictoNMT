import torch
import pytest
import tempfile
import os
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pictollms.models.encoders.eole_encoder import PictoEoleEncoder
from pictollms.models.schema.schema_inducer import SchemaInducer
from pictollms.models.schema.beam_search import CAsiBeamSearch
from pictollms.eval.metrics import evaluate_translations
from transformers import AutoTokenizer


# same mock decoder from the current train file, tracked on May 23
class MockModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    class MockDecoder:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
        
        def __call__(self, input_ids, encoder_hidden_states, encoder_attention_mask=None):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            
            return MockOutput(logits)
    
    def __init__(self, vocab_size):
        self.decoder = self.MockDecoder(vocab_size)

def test_encoder_forward():
    """Test encoder forward pass"""
    print("Testing encoder forward pass...")
    
    encoder = PictoEoleEncoder()
    
    # Test input
    batch_size, seq_len = 2, 4
    
    inputs = {
        'images': torch.randn(batch_size, seq_len, 3, 224, 224),
        'categories': torch.randint(0, 100, (batch_size, seq_len, 5)),
        'types': torch.randint(0, 10, (batch_size, seq_len))
    }
    
    encoder_final, memory_bank, lengths = encoder(inputs)
    
    assert encoder_final.shape == (batch_size, 512)
    assert memory_bank.shape == (batch_size, seq_len, 512)
    assert lengths.shape == (batch_size,)
    
    print("Encoder forward pass test passed")

def test_schema_inducer():
    """Test schema induction"""
    print("Testing schema inducer...")
    
    schema_inducer = SchemaInducer()
    
    batch_size, seq_len, hidden_size = 2, 4, 512
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    schema = schema_inducer(encoder_outputs, attention_mask)
    
    # Check schema components
    assert 'structure_type' in schema
    assert 'enhanced_repr' in schema
    assert 'determiner_logits' in schema
    assert schema['structure_type'].shape == (batch_size,)
    assert schema['enhanced_repr'].shape == (batch_size, seq_len, hidden_size)
    
    print("Schema inducer test passed")

def test_casi_beam_search():
    """Test CASI beam search"""
    print("Testing CASI beam search...")
    
    # Set up components
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
    beam_search = CAsiBeamSearch(beam_size=2, max_length=10)
    
    # Mock model
    model = MockModel(len(tokenizer))
    
    # Test inputs
    batch_size, src_len, hidden_size = 1, 4, 512
    encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
    
    # Mock schema
    schema = {
        'structure_type': torch.tensor([0]),
        'complexity_score': torch.tensor([[0.5]])
    }
    
    # Test beam search
    results = beam_search.search(
        model=model,
        encoder_outputs=encoder_outputs,
        schema=schema,
        tokenizer=tokenizer
    )
    
    assert len(results) == batch_size
    assert len(results[0]) > 1  # Should generate some tokens
    
    print("CASI beam search test passed")

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("Testing evaluation metrics...")
    
    predictions = [
        "le chat mange la souris",
        "la fille court dans le parc",
        "nous allons à l'école"
    ]
    
    references = [
        "le chat mange une souris",
        "la petite fille court dans le parc",
        "nous allons à l'école demain"
    ]
    
    metrics = evaluate_translations(predictions, references)
    
    # Check that all metrics are computed
    assert 'bleu' in metrics
    assert 'rouge_l' in metrics
    assert 'content_preservation' in metrics
    assert 'functional_word_accuracy' in metrics
    
    # Check reasonable ranges
    assert 0 <= metrics['bleu'] <= 100
    assert 0 <= metrics['rouge_l'] <= 100
    assert 0 <= metrics['content_preservation'] <= 100
    assert 0 <= metrics['functional_word_accuracy'] <= 100
    
    print("Evaluation metrics test passed")
    print(f"   BLEU: {metrics['bleu']:.2f}")
    print(f"   ROUGE-L: {metrics['rouge_l']:.2f}")
    print(f"   Content Preservation: {metrics['content_preservation']:.2f}")
    print(f"   Functional Word Accuracy: {metrics['functional_word_accuracy']:.2f}")

def test_end_to_end_pipeline():
    """Test complete pipeline"""
    print("Testing end-to-end pipeline...")
    
    # Set up components
    encoder = PictoEoleEncoder()
    schema_inducer = SchemaInducer()
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
    beam_search = CAsiBeamSearch(beam_size=2, max_length=10)
    model = MockModel(len(tokenizer))
    
    # Test input
    batch_size, seq_len = 1, 3
    
    inputs = {
        'images': torch.randn(batch_size, seq_len, 3, 224, 224),
        'categories': torch.randint(0, 100, (batch_size, seq_len, 5)),
        'types': torch.randint(0, 10, (batch_size, seq_len))
    }
    
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass through complete pipeline
    with torch.no_grad():
        # Encode
        encoder_final, memory_bank, lengths = encoder(inputs)
        
        # Generate schema
        schema = schema_inducer(memory_bank, attention_mask)
        
        # Decode with CASI
        results = beam_search.search(
            model=model,
            encoder_outputs=memory_bank,
            schema=schema,
            tokenizer=tokenizer,
            attention_mask=attention_mask
        )
        
        # Decode tokens to text
        decoded_texts = [tokenizer.decode(result, skip_special_tokens=True) for result in results]
    
    assert len(decoded_texts) == batch_size
    print(f"   Generated text: {decoded_texts[0]}")
    
    print("End-to-end pipeline test passed")

def test_memory_usage():
    """Test memory usage of complete pipeline"""
    print("Testing memory usage...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.reset_peak_memory_stats()
        
        # Set up components on GPU
        encoder = PictoEoleEncoder().to(device)
        schema_inducer = SchemaInducer().to(device)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            seq_len = 5
            
            inputs = {
                'images': torch.randn(batch_size, seq_len, 3, 224, 224).to(device),
                'categories': torch.randint(0, 100, (batch_size, seq_len, 5)).to(device),
                'types': torch.randint(0, 10, (batch_size, seq_len)).to(device)
            }
            
            attention_mask = torch.ones(batch_size, seq_len).to(device)
            
            with torch.no_grad():
                encoder_final, memory_bank, lengths = encoder(inputs)
                schema = schema_inducer(memory_bank, attention_mask)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"   Batch size {batch_size}: {memory_used:.1f} MB")
            
            # Clear cache
            torch.cuda.empty_cache()
        
        print("Memory usage test completed")
    else:
        print("GPU not available, skipping memory test")

if __name__ == "__main__":
    print("Running complete pipeline tests...\n")
    
    try:
        test_encoder_forward()
        test_schema_inducer()
        test_casi_beam_search()
        test_evaluation_metrics()
        test_end_to_end_pipeline()
        test_memory_usage()
        
        print("All tests passed! System is ready for training.")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)