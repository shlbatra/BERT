import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class BertConfig:
    vocab_size: int = 50261  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token + 1 PAD token + 1 CLS token + 1 SEP token + 1 MASK token
    hidden_size: int = 768 # embedding dimension (model capacity)
    num_hidden_layers: int = 12 # number of layers
    num_attention_heads: int = 12 # number of heads
    max_position_embeddings: int = 512 # max sequence length

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, type_vocab_size=3):
        super().__init__()
        # Create embedding lookup tables - convert discrete tokens to continuous vector representations
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)  # Technical: vocab_size x hidden_size lookup table; Conceptual: maps word tokens to semantic vectors
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)  # Technical: max_pos x hidden_size lookup; Conceptual: encodes positional information since transformers have no inherent position awareness
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)  # Technical: type_vocab x hidden_size lookup; Conceptual: distinguishes sentence A/B in tasks like NSP
        
        # Post-embedding normalization and regularization layers
        self.layer_norm = nn.LayerNorm(hidden_size)  # Technical: normalizes embeddings to mean=0, std=1; Conceptual: stabilizes training and improves convergence
        self.dropout = nn.Dropout(0.1)  # Technical: randomly zeros elements during training; Conceptual: prevents overfitting to specific embedding patterns
        
    def forward(self, input_ids, token_type_ids):
        # Generate position indices for the sequence
        seq_length = input_ids.size(1)  # Technical: extract sequence dimension; Conceptual: needed to create position encodings for variable-length sequences
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # Technical: creates [0,1,2,...,seq_len-1]; Conceptual: absolute position indices
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # Technical: (seq_length,) → (batch_size, seq_length); Conceptual: broadcast positions to match batch structure
        
        # Lookup embeddings for each input component
        token_embeddings = self.token_embeddings(input_ids)  # Technical: embedding lookup by token ID; Conceptual: converts discrete tokens to semantic vectors
        position_embeddings = self.position_embeddings(position_ids)  # Technical: embedding lookup by position; Conceptual: adds positional information to each token
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # Technical: embedding lookup by segment type; Conceptual: distinguishes different input segments
        
        # Combine all embedding types through addition
        embeddings = token_embeddings + position_embeddings + token_type_embeddings  # Technical: element-wise addition; Conceptual: combines semantic, positional, and segment information
        embeddings = self.layer_norm(embeddings)  # Technical: normalize to unit variance; Conceptual: ensures stable training dynamics
        embeddings = self.dropout(embeddings)  # Technical: applies dropout mask; Conceptual: regularization to prevent overfitting
        
        return embeddings  # Technical: return final embeddings; Conceptual: rich input representation ready for transformer layers
    

def scaled_dot_product_attention(Q, K, V, attn_mask=None):
    """Simplified scaled dot-product attention implementing: Attention(Q,K,V) = softmax(QK^T/√d_k)V"""
    
    # Extract key dimension for proper scaling - essential to prevent attention collapse
    d_k = Q.size(-1)  # Technical: gets last dimension size; Conceptual: scaling factor prevents gradient vanishing
    
    # Compute similarity scores between queries and keys
    scores = torch.matmul(Q, K.transpose(-2, -1))  # Technical: matrix multiplication QK^T; Conceptual: measures how much each query relates to each key
    scores = scores / math.sqrt(d_k)  # Technical: element-wise division by √d_k; Conceptual: prevents large dot products that cause softmax saturation and gradient vanishing
    
    # Apply attention masking to ignore irrelevant positions
    if attn_mask is not None:  # Technical: check if mask provided; Conceptual: allows selective attention (ignore padding, future tokens, etc.)
        scores = scores + attn_mask  # Technical: adds pre-computed additive mask; Conceptual: large negative values make softmax output ~0 for masked positions
    
    # Convert raw scores to probability distribution
    attn_weights = F.softmax(scores, dim=-1)  # Technical: applies softmax along last dimension; Conceptual: creates probability distribution (weights sum to 1)
    
    # Compute final output as weighted combination of values
    context = torch.matmul(attn_weights, V)  # Technical: matrix multiplication of weights and values; Conceptual: weighted average of all value vectors based on attention
    
    return context, attn_weights  # Technical: return tuple; Conceptual: context for next layer, weights for analysis/visualization 

class BertMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size  # Store hidden size for tensor reshaping operations
        self.num_attention_heads = num_attention_heads  # Number of parallel attention heads
        self.attention_head_size = hidden_size // num_attention_heads  # Size of each attention head
        
        # Linear projections for Q, K, V - transform input to query/key/value representations
        self.query = nn.Linear(hidden_size, hidden_size)  # Projects input to query space
        self.key = nn.Linear(hidden_size, hidden_size)    # Projects input to key space  
        self.value = nn.Linear(hidden_size, hidden_size)  # Projects input to value space
        
        # Output projection and normalization layers - must be in __init__ for proper parameter registration
        self.output_proj = nn.Linear(hidden_size, hidden_size)  # Final linear transformation after attention
        self.layer_norm = nn.LayerNorm(hidden_size)  # Normalizes residual connection for stable training
        self.dropout = nn.Dropout(0.1)  # Prevents overfitting by randomly zeroing elements
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.size()[:2]  # Extract dimensions for tensor reshaping
        
        # Linear projections: transform input to Q/K/V representations
        Q = self.query(hidden_states)  # Query: "what am I looking for?" 
        K = self.key(hidden_states)    # Key: "what do I contain?"
        V = self.value(hidden_states)  # Value: "what information do I carry?"
        
        # Reshape for multi-head attention: split hidden_size into num_heads × head_size
        Q = Q.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)  # (B,S,H) → (B,NH,S,HS)
        K = K.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)  # Allows parallel attention computation
        V = V.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)  # Each head learns different aspects
        
        # Attention mask is already in correct format from Bert.forward() - no additional processing needed
        
        # Apply scaled dot-product attention across all heads in parallel
        context, attn_weights = scaled_dot_product_attention(Q, K, V, attention_mask)  # Core attention computation
        
        # Reshape back to original dimensions: concatenate all attention heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)  # (B,NH,S,HS) → (B,S,H)
        
        # Apply output transformations with residual connection (critical for deep network training)
        output = self.output_proj(context)  # Final linear transformation to mix head outputs
        output = self.dropout(output)  # Apply dropout for regularization
        output = self.layer_norm(output + hidden_states)  # Add & Norm: residual connection + normalization
        
        return output  # Return final attention output


class BertPoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Two-layer MLP with 4x expansion (standard BERT architecture)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)  # Expand: projects to 4x larger dimension for increased capacity
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)  # Contract: projects back to original dimension
        self.dropout = nn.Dropout(0.1)  # Regularization to prevent overfitting
        
    def forward(self, x):
        # Position-wise feed-forward: applies same transformation to each position independently
        x = self.fc1(x)  # Linear expansion: (batch_size, seq_len, hidden_size) → (batch_size, seq_len, hidden_size*4)
        x = F.gelu(x)    # GELU activation: smooth activation function, works better than ReLU for transformers
        x = self.dropout(x)  # Apply dropout for regularization during training
        x = self.fc2(x)  # Linear contraction: (batch_size, seq_len, hidden_size*4) → (batch_size, seq_len, hidden_size)
        return x  # Return transformed representation with original dimensions preserved



class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        # Multi-head self-attention mechanism - core of transformer architecture
        self.self_attention = BertMultiHeadAttention(hidden_size, num_attention_heads)  # Technical: attention with proper parameters; Conceptual: allows model to relate different positions in sequence
        
        # Position-wise feed-forward network - processes each position independently
        self.feed_forward = BertPoswiseFeedForwardNet(hidden_size)  # Technical: MLP with expansion/contraction; Conceptual: adds non-linearity and position-wise transformation
        
        # Layer normalization for feed-forward output
        self.layer_norm = nn.LayerNorm(hidden_size)  # Technical: normalizes FFN output; Conceptual: stabilizes training for deep networks
        self.dropout = nn.Dropout(0.1)  # Technical: random zeroing during training; Conceptual: prevents overfitting in FFN pathway
        
    def forward(self, hidden_states, attention_mask=None):
        # Apply self-attention with residual connection and layer norm (already built into attention module)
        attn_output = self.self_attention(hidden_states, attention_mask)  # Technical: self-attention with Add&Norm; Conceptual: relates all positions in sequence
        
        # Apply feed-forward network with residual connection
        ffn_output = self.feed_forward(attn_output)  # Technical: position-wise MLP transformation; Conceptual: adds non-linear processing capacity
        ffn_output = self.dropout(ffn_output)  # Technical: applies dropout for regularization; Conceptual: prevents overfitting during training
        output = self.layer_norm(ffn_output + attn_output)  # Technical: Add&Norm (residual + normalization); Conceptual: enables deep network training and gradient flow
        
        return output  # Technical: return transformed representation; Conceptual: enriched sequence representation for next layer


class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store config for easy access
        self.config = config
        
        # Input embedding layer - converts tokens to rich representations
        self.embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.max_position_embeddings)  # Technical: embedding with proper parameters; Conceptual: converts discrete tokens to continuous vectors
        
        # Stack of transformer encoder layers - core processing units
        self.encoder_layers = nn.ModuleList([
            BertEncoderLayer(config.hidden_size, config.num_attention_heads) 
            for _ in range(config.num_hidden_layers)
        ])  # Technical: creates list of encoder layers; Conceptual: stacks multiple attention+FFN layers for deep representation learning
        
        # NSP (Next Sentence Prediction) components
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)  # Technical: linear projection of [CLS] token; Conceptual: creates sequence-level representation for NSP
        self.pooler_activation = nn.Tanh()  # Technical: tanh activation function; Conceptual: bounds pooled representation
        self.nsp_classifier = nn.Linear(config.hidden_size, 2)  # Technical: binary classification layer; Conceptual: predicts if sentence B follows A
        
        # MLM (Masked Language Modeling) components  
        self.mlm_transform = nn.Linear(config.hidden_size, config.hidden_size)  # Technical: linear transformation; Conceptual: projects hidden states for token prediction
        self.mlm_norm = nn.LayerNorm(config.hidden_size)  # Technical: layer normalization; Conceptual: stabilizes MLM predictions
        self.mlm_activation = F.gelu  # Technical: GELU activation; Conceptual: non-linearity for MLM head
        
        # MLM decoder with weight sharing
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # Technical: vocab prediction layer; Conceptual: predicts masked tokens
        self.mlm_decoder.weight = self.embeddings.token_embeddings.weight  # Technical: tie weights; Conceptual: shared embedding/output reduces parameters
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))  # Technical: learnable bias; Conceptual: improves MLM prediction accuracy
        
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        # Convert input tokens to embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)  # Technical: token+position+segment embeddings; Conceptual: rich input representation
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0)  # Technical: mask padding tokens (assuming 0 = PAD); Conceptual: prevents attention to padding
        
        # Convert mask to attention format
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Technical: expand dims for broadcasting; Conceptual: shape for multi-head attention
        extended_attention_mask = (~extended_attention_mask) * -10000.0  # Technical: convert to additive mask; Conceptual: large negative values → ~0 attention weights
        
        # Pass through all encoder layers sequentially
        for encoder_layer in self.encoder_layers:  # Technical: sequential layer application; Conceptual: progressive refinement of representations
            hidden_states = encoder_layer(hidden_states, extended_attention_mask)  # Technical: attention+FFN transformation; Conceptual: builds increasingly abstract representations
        
        # NSP: Pool sequence representation from [CLS] token
        pooled_output = self.pooler(hidden_states[:, 0])  # Technical: linear transform of first token; Conceptual: [CLS] token aggregates sequence information
        pooled_output = self.pooler_activation(pooled_output)  # Technical: apply tanh activation; Conceptual: bounded representation for classification
        nsp_logits = self.nsp_classifier(pooled_output)  # Technical: binary classification; Conceptual: predicts sentence relationship
        
        # MLM: Transform hidden states for token prediction
        mlm_hidden = self.mlm_transform(hidden_states)  # Technical: linear transformation; Conceptual: projects to MLM prediction space
        mlm_hidden = self.mlm_activation(mlm_hidden)  # Technical: GELU activation; Conceptual: adds non-linearity for better predictions
        mlm_hidden = self.mlm_norm(mlm_hidden)  # Technical: layer normalization; Conceptual: stabilizes MLM head training
        mlm_logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias  # Technical: vocab prediction + bias; Conceptual: final token probabilities
        
        return {
            'nsp_logits': nsp_logits,            # Technical: NSP classification scores; Conceptual: sentence relationship prediction
            'mlm_logits': mlm_logits             # Technical: MLM prediction scores; Conceptual: masked token predictions
        }