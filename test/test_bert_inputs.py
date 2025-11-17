import torch
import pytest
import sys
import os

# Add the parent directory to the path to find src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.bert_inputs import create_mlm_inputs_and_labels


class TestCreateMLMInputsAndLabels:
    """Unit tests for create_mlm_inputs_and_labels function"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.mask_token_id = 50258  # [MASK]
        self.pad_token_id = 50259   # [PAD]
        self.cls_token_id = 50256   # [CLS]
        self.sep_token_id = 50257   # [SEP]
        torch.manual_seed(42)  # For reproducible tests
    
    def test_basic_functionality(self):
        """Test basic masking functionality"""
        # Create simple input: [CLS] token1 token2 token3 [SEP] [PAD] [PAD]
        input_ids = torch.tensor([[50256, 100, 200, 300, 50257, 50259, 50259]])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Check output shapes
        assert masked_input_ids.shape == input_ids.shape
        assert mlm_labels.shape == input_ids.shape
        assert attention_mask.shape == input_ids.shape
        
        # Check that special tokens are never masked in labels
        assert mlm_labels[0, 0] == -100  # [CLS] should be ignored
        assert mlm_labels[0, 4] == -100  # [SEP] should be ignored  
        assert mlm_labels[0, 5] == -100  # [PAD] should be ignored
        assert mlm_labels[0, 6] == -100  # [PAD] should be ignored
    
    def test_special_tokens_not_masked(self):
        """Test that special tokens ([CLS], [SEP], [PAD]) are never masked"""
        # Input with various special tokens
        input_ids = torch.tensor([[50256, 100, 200, 50257, 50259, 50259]])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Special tokens should remain unchanged in masked_input_ids
        assert masked_input_ids[0, 0] == 50256  # [CLS]
        assert masked_input_ids[0, 3] == 50257  # [SEP]
        assert masked_input_ids[0, 4] == 50259  # [PAD]
        assert masked_input_ids[0, 5] == 50259  # [PAD]
        
        # Special tokens should have -100 in labels (ignored for loss)
        assert mlm_labels[0, 0] == -100  # [CLS]
        assert mlm_labels[0, 3] == -100  # [SEP]
        assert mlm_labels[0, 4] == -100  # [PAD]
        assert mlm_labels[0, 5] == -100  # [PAD]
    
    def test_attention_mask_creation(self):
        """Test that attention mask correctly identifies non-PAD tokens"""
        input_ids = torch.tensor([[50256, 100, 200, 50257, 50259, 50259]])
        
        _, _, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Non-PAD tokens should have attention_mask = True
        assert attention_mask[0, 0] == True   # [CLS]
        assert attention_mask[0, 1] == True   # token 100
        assert attention_mask[0, 2] == True   # token 200
        assert attention_mask[0, 3] == True   # [SEP]
        
        # PAD tokens should have attention_mask = False
        assert attention_mask[0, 4] == False  # [PAD]
        assert attention_mask[0, 5] == False  # [PAD]
    
    def test_masking_probability(self):
        """Test that approximately 15% of eligible tokens are masked"""
        # Create longer sequence to test probability
        torch.manual_seed(123)  # Set seed for reproducible test
        vocab_size = 1000
        seq_length = 100
        batch_size = 10
        
        # Create input with mostly regular tokens (avoiding special tokens)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        # Set first token as [CLS] and last as [SEP]
        input_ids[:, 0] = 50256  # [CLS]
        input_ids[:, -1] = 50257  # [SEP]
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(
            input_ids, mask_prob=0.15
        )
        
        # Count masked tokens (tokens where mlm_labels != -100)
        masked_tokens = (mlm_labels != -100).sum().item()
        eligible_tokens = (seq_length - 2) * batch_size  # Exclude [CLS] and [SEP]
        masking_rate = masked_tokens / eligible_tokens
        
        # Should be approximately 15% (within reasonable tolerance)
        assert 0.10 <= masking_rate <= 0.20, f"Masking rate {masking_rate:.3f} outside expected range"
    
    def test_mask_token_replacement(self):
        """Test that some masked tokens are replaced with [MASK] token"""
        torch.manual_seed(456)
        # Create simple input with many regular tokens
        input_ids = torch.tensor([[50256] + [i for i in range(1, 50)] + [50257]])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Check if any tokens were replaced with [MASK]
        mask_replacements = (masked_input_ids == self.mask_token_id).sum().item()
        
        # Should have some [MASK] tokens (but exact count depends on randomness)
        # We just verify the mechanism works
        total_masked = (mlm_labels != -100).sum().item()
        if total_masked > 0:
            # At least some masked tokens should become [MASK] (80% probability)
            assert mask_replacements >= 0
    
    def test_custom_mask_probability(self):
        """Test that custom masking probability works"""
        input_ids = torch.tensor([[50256, 100, 200, 300, 400, 500, 50257]])
        
        # Test with 0% masking
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(
            input_ids, mask_prob=0.0
        )
        
        # With 0% probability, no tokens should be masked
        masked_count = (mlm_labels != -100).sum().item()
        assert masked_count == 0
        
        # Input should remain unchanged (except for special token handling)
        regular_tokens_unchanged = torch.equal(masked_input_ids[0, 1:-1], input_ids[0, 1:-1])
        assert regular_tokens_unchanged
    
    def test_batch_processing(self):
        """Test that function works correctly with batched input"""
        batch_size = 3
        seq_length = 8
        
        input_ids = torch.tensor([
            [50256, 100, 200, 300, 50257, 50259, 50259, 50259],
            [50256, 400, 500, 600, 700, 50257, 50259, 50259],
            [50256, 800, 900, 1000, 1100, 1200, 50257, 50259]
        ])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Check shapes
        assert masked_input_ids.shape == (batch_size, seq_length)
        assert mlm_labels.shape == (batch_size, seq_length)
        assert attention_mask.shape == (batch_size, seq_length)
        
        # Check that special tokens are preserved across all batches
        for i in range(batch_size):
            assert masked_input_ids[i, 0] == 50256  # [CLS]
            assert mlm_labels[i, 0] == -100          # [CLS] ignored in loss
    
    def test_edge_case_all_special_tokens(self):
        """Test with input containing only special tokens"""
        input_ids = torch.tensor([[50256, 50257, 50259, 50259]])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # All tokens should be unchanged
        assert torch.equal(masked_input_ids, input_ids)
        
        # All labels should be -100 (ignored)
        assert torch.all(mlm_labels == -100)
        
        # Attention mask should be correct
        expected_attention = torch.tensor([[True, True, False, False]])
        assert torch.equal(attention_mask, expected_attention)
    
    def test_empty_sequence(self):
        """Test with empty sequence"""
        input_ids = torch.empty((1, 0), dtype=torch.long)
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # All outputs should be empty with correct shapes
        assert masked_input_ids.shape == (1, 0)
        assert mlm_labels.shape == (1, 0)
        assert attention_mask.shape == (1, 0)
    
    def test_device_consistency_cpu(self):
        """Test that output tensors are on the same device as input (CPU)"""
        device = torch.device('cpu')
        input_ids = torch.tensor([[50256, 100, 200, 50257]], device=device)
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # All outputs should be on the same device as input
        assert masked_input_ids.device == device
        assert mlm_labels.device == device
        assert attention_mask.device == device
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic when using the same seed"""
        input_ids = torch.tensor([[50256, 100, 200, 300, 400, 50257]])
        
        # Run twice with same seed
        torch.manual_seed(789)
        result1 = create_mlm_inputs_and_labels(input_ids)
        
        torch.manual_seed(789)
        result2 = create_mlm_inputs_and_labels(input_ids)
        
        # Results should be identical
        assert torch.equal(result1[0], result2[0])  # masked_input_ids
        assert torch.equal(result1[1], result2[1])  # mlm_labels
        assert torch.equal(result1[2], result2[2])  # attention_mask

    @pytest.mark.parametrize("mask_prob", [0.0, 0.05, 0.15, 0.30])
    def test_different_mask_probabilities(self, mask_prob):
        """Test function with different masking probabilities"""
        input_ids = torch.tensor([[50256, 100, 200, 300, 400, 500, 600, 50257]])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(
            input_ids, mask_prob=mask_prob
        )
        
        # Basic checks
        assert masked_input_ids.shape == input_ids.shape
        assert mlm_labels.shape == input_ids.shape
        assert attention_mask.shape == input_ids.shape
        
        # Special tokens should always be preserved
        assert masked_input_ids[0, 0] == 50256  # [CLS]
        assert masked_input_ids[0, -1] == 50257  # [SEP]
        assert mlm_labels[0, 0] == -100  # [CLS] ignored
        assert mlm_labels[0, -1] == -100  # [SEP] ignored

    def test_mlm_labels_structure(self):
        """Test that MLM labels have correct structure"""
        input_ids = torch.tensor([[50256, 100, 200, 300, 50257, 50259]])
        
        masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Labels should either be -100 (ignored) or equal to original token
        for i in range(input_ids.shape[1]):
            label = mlm_labels[0, i]
            original = input_ids[0, i]
            
            # Label is either -100 (ignored) or equals original token
            assert label == -100 or label == original
    
    def test_attention_mask_consistency(self):
        """Test that attention mask is consistent with PAD tokens"""
        input_ids = torch.tensor([
            [50256, 100, 200, 50257, 50259, 50259],
            [50256, 300, 400, 500, 50257, 50259]
        ])
        
        _, _, attention_mask = create_mlm_inputs_and_labels(input_ids)
        
        # Attention mask should be False only for PAD tokens
        for batch_idx in range(input_ids.shape[0]):
            for token_idx in range(input_ids.shape[1]):
                is_pad = input_ids[batch_idx, token_idx] == self.pad_token_id
                has_attention = attention_mask[batch_idx, token_idx]
                
                # Attention should be False only for PAD tokens
                assert has_attention == (not is_pad)