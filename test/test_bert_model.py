#!/usr/bin/env python3
"""
Comprehensive test suite for bert_model.py components
"""

import torch
import torch.nn as nn
import sys

sys.path.append('src')

from src.bert_model import (
    BertEmbeddings,
    BertSelfAttention,
    BertSelfOutput,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayer,
    BertEncoder,
    BertPooler,
    BertModel,
    BertForPreTraining
)


class TestBertEmbeddings:
    def test_init(self):
        vocab_size, hidden_size = 1000, 768
        embeddings = BertEmbeddings(vocab_size, hidden_size)
        
        assert embeddings.token_embeddings.num_embeddings == vocab_size
        assert embeddings.token_embeddings.embedding_dim == hidden_size
        assert embeddings.position_embeddings.num_embeddings == 512  # default max_position_embeddings
        assert embeddings.token_type_embeddings.num_embeddings == 3  # default type_vocab_size

    def test_forward(self):
        vocab_size, hidden_size = 1000, 768
        batch_size, seq_length = 2, 10
        embeddings = BertEmbeddings(vocab_size, hidden_size)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 3, (batch_size, seq_length))
        
        output = embeddings(input_ids, token_type_ids)
        
        assert output.shape == (batch_size, seq_length, hidden_size)
        assert output.dtype == torch.float32

    def test_position_ids_generation(self):
        vocab_size, hidden_size = 1000, 768
        batch_size, seq_length = 2, 5
        embeddings = BertEmbeddings(vocab_size, hidden_size)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
        
        output = embeddings(input_ids, token_type_ids)
        
        assert output is not None
        assert not torch.isnan(output).any()


class TestBertSelfAttention:
    def test_init(self):
        hidden_size, num_heads = 768, 12
        attention = BertSelfAttention(hidden_size, num_heads)
        
        assert attention.num_attention_heads == num_heads
        assert attention.attention_head_size == hidden_size // num_heads
        assert attention.all_head_size == hidden_size

    def test_transpose_for_scores(self):
        hidden_size, num_heads = 768, 12
        batch_size, seq_length = 2, 10
        attention = BertSelfAttention(hidden_size, num_heads)
        
        x = torch.randn(batch_size, seq_length, hidden_size)
        transposed = attention.transpose_for_scores(x)
        
        expected_shape = (batch_size, num_heads, seq_length, hidden_size // num_heads)
        assert transposed.shape == expected_shape

    def test_forward(self):
        hidden_size, num_heads = 768, 12
        batch_size, seq_length = 2, 10
        attention = BertSelfAttention(hidden_size, num_heads)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        output = attention(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)

    def test_forward_with_attention_mask(self):
        hidden_size, num_heads = 768, 12
        batch_size, seq_length = 2, 10
        attention = BertSelfAttention(hidden_size, num_heads)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.zeros(batch_size, num_heads, seq_length, seq_length)
        attention_mask[:, :, :, 5:] = -10000.0  # Mask last 5 positions
        
        output = attention(hidden_states, attention_mask)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestBertSelfOutput:
    def test_init(self):
        hidden_size = 768
        output_layer = BertSelfOutput(hidden_size)
        
        assert output_layer.dense.in_features == hidden_size
        assert output_layer.dense.out_features == hidden_size

    def test_forward(self):
        hidden_size = 768
        batch_size, seq_length = 2, 10
        output_layer = BertSelfOutput(hidden_size)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        input_tensor = torch.randn(batch_size, seq_length, hidden_size)
        
        output = output_layer(hidden_states, input_tensor)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestBertAttention:
    def test_init(self):
        hidden_size, num_heads = 768, 12
        attention = BertAttention(hidden_size, num_heads)
        
        assert isinstance(attention.self, BertSelfAttention)
        assert isinstance(attention.output, BertSelfOutput)

    def test_forward(self):
        hidden_size, num_heads = 768, 12
        batch_size, seq_length = 2, 10
        attention = BertAttention(hidden_size, num_heads)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        output = attention(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestBertIntermediate:
    def test_init(self):
        hidden_size, intermediate_size = 768, 3072
        intermediate = BertIntermediate(hidden_size, intermediate_size)
        
        assert intermediate.dense.in_features == hidden_size
        assert intermediate.dense.out_features == intermediate_size

    def test_forward(self):
        hidden_size, intermediate_size = 768, 3072
        batch_size, seq_length = 2, 10
        intermediate = BertIntermediate(hidden_size, intermediate_size)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        output = intermediate(hidden_states)
        
        assert output.shape == (batch_size, seq_length, intermediate_size)
        assert torch.all(output >= 0)  # GELU should produce non-negative values for positive inputs


class TestBertOutput:
    def test_init(self):
        intermediate_size, hidden_size = 3072, 768
        output_layer = BertOutput(intermediate_size, hidden_size)
        
        assert output_layer.dense.in_features == intermediate_size
        assert output_layer.dense.out_features == hidden_size

    def test_forward(self):
        intermediate_size, hidden_size = 3072, 768
        batch_size, seq_length = 2, 10
        output_layer = BertOutput(intermediate_size, hidden_size)
        
        hidden_states = torch.randn(batch_size, seq_length, intermediate_size)
        input_tensor = torch.randn(batch_size, seq_length, hidden_size)
        
        output = output_layer(hidden_states, input_tensor)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestBertLayer:
    def test_init(self):
        hidden_size, num_heads, intermediate_size = 768, 12, 3072
        layer = BertLayer(hidden_size, num_heads, intermediate_size)
        
        assert isinstance(layer.attention, BertAttention)
        assert isinstance(layer.intermediate, BertIntermediate)
        assert isinstance(layer.output, BertOutput)

    def test_forward(self):
        hidden_size, num_heads, intermediate_size = 768, 12, 3072
        batch_size, seq_length = 2, 10
        layer = BertLayer(hidden_size, num_heads, intermediate_size)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        output = layer(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestBertEncoder:
    def test_init(self):
        num_layers, hidden_size, num_heads, intermediate_size = 12, 768, 12, 3072
        encoder = BertEncoder(num_layers, hidden_size, num_heads, intermediate_size)
        
        assert len(encoder.layers) == num_layers
        assert all(isinstance(layer, BertLayer) for layer in encoder.layers)

    def test_forward(self):
        num_layers, hidden_size, num_heads, intermediate_size = 6, 768, 12, 3072
        batch_size, seq_length = 2, 10
        encoder = BertEncoder(num_layers, hidden_size, num_heads, intermediate_size)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        output = encoder(hidden_states)
        
        assert output.shape == (batch_size, seq_length, hidden_size)


class TestBertPooler:
    def test_init(self):
        hidden_size = 768
        pooler = BertPooler(hidden_size)
        
        assert pooler.dense.in_features == hidden_size
        assert pooler.dense.out_features == hidden_size

    def test_forward(self):
        hidden_size = 768
        batch_size, seq_length = 2, 10
        pooler = BertPooler(hidden_size)
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        output = pooler(hidden_states)
        
        assert output.shape == (batch_size, hidden_size)
        assert torch.all(torch.abs(output) <= 1.0)  # tanh output should be in [-1, 1]


class TestBertModel:
    def test_init(self):
        model = BertModel(vocab_size=1000, hidden_size=768)
        
        assert isinstance(model.embeddings, BertEmbeddings)
        assert isinstance(model.encoder, BertEncoder)
        assert isinstance(model.pooler, BertPooler)

    def test_forward(self):
        vocab_size, hidden_size = 1000, 768
        batch_size, seq_length = 2, 10
        model = BertModel(vocab_size, hidden_size, num_hidden_layers=6)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 2, (batch_size, seq_length))
        
        sequence_output, pooled_output = model(input_ids, token_type_ids)
        
        assert sequence_output.shape == (batch_size, seq_length, hidden_size)
        assert pooled_output.shape == (batch_size, hidden_size)

    def test_forward_with_attention_mask(self):
        vocab_size, hidden_size = 1000, 768
        batch_size, seq_length = 2, 10
        model = BertModel(vocab_size, hidden_size, num_hidden_layers=3)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 2, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, 5:] = 0  # Mask last 5 positions
        
        sequence_output, pooled_output = model(input_ids, token_type_ids, attention_mask)
        
        assert sequence_output.shape == (batch_size, seq_length, hidden_size)
        assert pooled_output.shape == (batch_size, hidden_size)

    def test_attention_mask_conversion(self):
        vocab_size, hidden_size = 1000, 768
        batch_size, seq_length = 2, 5
        model = BertModel(vocab_size, hidden_size, num_hidden_layers=2)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        
        sequence_output, pooled_output = model(input_ids, token_type_ids, attention_mask)
        
        assert not torch.isnan(sequence_output).any()
        assert not torch.isnan(pooled_output).any()


class TestBertForPreTraining:
    def test_init(self):
        vocab_size = 1000
        model = BertForPreTraining(vocab_size)
        
        assert isinstance(model.bert, BertModel)
        assert model.mlm_head.out_features == vocab_size
        assert model.nsp_head.out_features == 2

    def test_forward(self):
        vocab_size, hidden_size = 1000, 768
        batch_size, seq_length = 2, 10
        model = BertForPreTraining(vocab_size, hidden_size, num_hidden_layers=3)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 2, (batch_size, seq_length))
        
        mlm_scores, nsp_scores = model(input_ids, token_type_ids)
        
        assert mlm_scores.shape == (batch_size, seq_length, vocab_size)
        assert nsp_scores.shape == (batch_size, 2)

    def test_training_step_simulation(self):
        vocab_size, hidden_size = 1000, 256
        batch_size, seq_length = 2, 8
        model = BertForPreTraining(vocab_size, hidden_size, num_hidden_layers=2, num_attention_heads=4, intermediate_size=512)
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 2, (batch_size, seq_length))
        mlm_labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        nsp_labels = torch.randint(0, 2, (batch_size,))
        
        model.train()
        mlm_scores, nsp_scores = model(input_ids, token_type_ids)
        
        mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_scores.view(-1, vocab_size), mlm_labels.view(-1))
        nsp_loss = nn.CrossEntropyLoss()(nsp_scores, nsp_labels)
        
        total_loss = mlm_loss + nsp_loss
        
        assert not torch.isnan(total_loss)
        assert total_loss.item() > 0


class TestModelDimensions:
    def test_dimension_consistency(self):
        """Test that dimensions are consistent across all components"""
        vocab_size, hidden_size, num_heads = 1000, 768, 12
        intermediate_size = 3072
        num_layers = 6
        
        model = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size, 
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size
        )
        
        assert hidden_size % num_heads == 0  # Must be divisible for multi-head attention
        
        batch_size, seq_length = 2, 16
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 2, (batch_size, seq_length))
        
        sequence_output, pooled_output = model(input_ids, token_type_ids)
        
        assert sequence_output.shape == (batch_size, seq_length, hidden_size)
        assert pooled_output.shape == (batch_size, hidden_size)

    def test_parameter_counts(self):
        """Test parameter counts for different model sizes"""
        small_model = BertModel(vocab_size=1000, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)
        large_model = BertModel(vocab_size=1000, hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048)
        
        small_params = sum(p.numel() for p in small_model.parameters())
        large_params = sum(p.numel() for p in large_model.parameters())
        
        assert large_params > small_params


def run_tests():
    """Run all tests and report results"""
    import traceback
    
    test_classes = [
        TestBertEmbeddings,
        TestBertSelfAttention, 
        TestBertSelfOutput,
        TestBertAttention,
        TestBertIntermediate,
        TestBertOutput,
        TestBertLayer,
        TestBertEncoder,
        TestBertPooler,
        TestBertModel,
        TestBertForPreTraining,
        TestModelDimensions
    ]
    
    total_tests = 0
    passed_tests = 0
    
    print("🧪 Running BERT Model Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n📋 {class_name}")
        print("-" * 40)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"  ✅ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! BERT model is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)