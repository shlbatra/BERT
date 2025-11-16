#!/usr/bin/env python3
"""
Test script to verify BERT pipeline components work correctly
"""

import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append('src')

def test_bert_model():
    """Test BERT model architecture"""
    print("Testing BERT model architecture...")
    
    try:
        from src.bert_model import BertForPreTraining
        
        # Create small model for testing
        model = BertForPreTraining(
            vocab_size=50260,
            hidden_size=256,  # Smaller for testing
            num_hidden_layers=4,  # Fewer layers
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 50260, (batch_size, seq_len))
        segment_ids = torch.randint(0, 2, (batch_size, seq_len))
        
        mlm_scores, nsp_scores = model(input_ids, segment_ids)
        
        print(f"✓ Model forward pass successful")
        print(f"  MLM scores shape: {mlm_scores.shape}")
        print(f"  NSP scores shape: {nsp_scores.shape}")
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ BERT model test failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing functions"""
    print("\nTesting data preprocessing...")
    
    try:
        from src.data_scripts.fineweb import split_into_sentences, create_sentence_pairs, tokenize_bert_pair
        
        # Test sentence splitting
        test_doc = {
            "text": "This is the first sentence. This is the second sentence! And this is the third? Final sentence here."
        }
        
        sentences = split_into_sentences(test_doc["text"])
        print(f"✓ Sentence splitting: {len(sentences)} sentences")
        
        # Test sentence pair creation
        pairs = create_sentence_pairs(test_doc)
        print(f"✓ Sentence pairs: {len(pairs)} pairs created")
        
        # Test tokenization
        if pairs:
            tokenized = tokenize_bert_pair(pairs[0])
            print(f"✓ Tokenization successful")
            print(f"  Input shape: {tokenized['tokens'].shape}")
            print(f"  Segment shape: {tokenized['segment_ids'].shape}")
            print(f"  NSP label: {tokenized['is_next']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        return False


def test_mlm_function():
    """Test MLM masking function"""
    print("\nTesting MLM masking...")
    
    try:
        from src.train_bert import create_mlm_inputs_and_labels
        
        # Create test input
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(100, 50256, (batch_size, seq_len))  # Avoid special tokens
        
        masked_input_ids, mlm_labels = create_mlm_inputs_and_labels(input_ids)
        
        print(f"✓ MLM masking successful")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Masked input shape: {masked_input_ids.shape}")
        print(f"  Labels shape: {mlm_labels.shape}")
        
        # Check that some tokens were masked
        mask_token_id = 50258
        num_masked = (masked_input_ids == mask_token_id).sum().item()
        print(f"  Tokens masked: {num_masked}")
        
        return True
        
    except Exception as e:
        print(f"✗ MLM masking test failed: {e}")
        return False


def test_compatibility():
    """Test compatibility between components"""
    print("\nTesting component compatibility...")
    
    try:
        from src.bert_model import BertForPreTraining
        from src.train_bert import create_mlm_inputs_and_labels
        import torch.nn.functional as F
        
        # Create model
        model = BertForPreTraining(
            vocab_size=50260,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512
        )
        
        # Create test data
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(100, 50256, (batch_size, seq_len))
        segment_ids = torch.randint(0, 2, (batch_size, seq_len))
        nsp_labels = torch.randint(0, 2, (batch_size,))
        
        # Test MLM
        masked_input_ids, mlm_labels = create_mlm_inputs_and_labels(input_ids)
        
        # Test forward pass
        mlm_scores, nsp_scores = model(masked_input_ids, segment_ids)
        
        # Test loss computation
        mlm_loss = F.cross_entropy(mlm_scores.view(-1, 50260), mlm_labels.view(-1), ignore_index=-100)
        nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)
        total_loss = mlm_loss + nsp_loss
        
        print(f"✓ End-to-end compatibility successful")
        print(f"  MLM loss: {mlm_loss.item():.4f}")
        print(f"  NSP loss: {nsp_loss.item():.4f}")
        print(f"  Total loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("🤖 Testing BERT Pipeline Components")
    print("=" * 50)
    
    tests = [
        test_bert_model,
        test_data_preprocessing, 
        test_mlm_function,
        test_compatibility
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! BERT pipeline is ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")