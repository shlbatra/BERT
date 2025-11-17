# BERT Implementation from Scratch

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

This repository implements BERT from scratch using PyTorch, including training infrastructure and evaluation on downstream tasks.

## Architecture Overview

### BERT Encoder Components

The BERT model consists of several key components implemented in `src/training/bert.py`:

#### 1. **BertEmbeddings**
- **Token Embeddings**: Maps vocabulary tokens to dense vectors (vocab_size × hidden_size)
- **Position Embeddings**: Encodes positional information (max_seq_len × hidden_size) 
- **Token Type Embeddings**: Distinguishes sentence A/B for NSP tasks (type_vocab_size × hidden_size)
- **Layer Normalization + Dropout**: Stabilizes training and prevents overfitting

#### 2. **Scaled Dot-Product Attention**
```python
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```
- Computes attention weights between query and key vectors
- Scales by √d_k to prevent gradient vanishing
- Supports attention masking for padding tokens

#### 3. **BertMultiHeadAttention** 
- Parallel attention heads (default: 6 heads for compact model)
- Each head learns different representation aspects
- Concatenates outputs and applies linear projection
- Includes residual connection and layer normalization

#### 4. **BertPoswiseFeedForwardNet**
- Two-layer MLP with GELU activation
- Expands to 4×hidden_size then contracts back
- Position-wise processing (same transformation per position)

#### 5. **BertEncoderLayer**
- Combines multi-head attention + feed-forward network
- Residual connections around both sub-layers
- Layer normalization for training stability

#### 6. **Full BERT Model**
- Stack of 6 encoder layers (configurable)
- MLM (Masked Language Model) prediction head
- NSP (Next Sentence Prediction) classification head
- Weight sharing between embedding and output layers

## Model Configuration

Current compact configuration in `BertConfig`:
```python
vocab_size: 50261          # Vocabulary size
hidden_size: 384           # Model dimension (compact vs 768 standard)
num_hidden_layers: 6       # Number of transformer layers (vs 12 standard) 
num_attention_heads: 6     # Number of attention heads (vs 12 standard)
max_position_embeddings: 512  # Maximum sequence length
```

## Training Setup

### 1. **Data Processing**
- Uses `DataLoaderLite` for efficient batch loading from GCP storage
- Implements MLM masking with 15% probability:
  - 80% replaced with [MASK] token
  - 10% replaced with random tokens  
  - 10% kept unchanged
- Creates attention masks for padding tokens

### 2. **Training Objectives**

#### Masked Language Modeling (MLM)
```python
# Implemented in create_mlm_inputs_and_labels()
masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
mlm_loss = F.cross_entropy(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1), ignore_index=-100)
```

#### Next Sentence Prediction (NSP) 
```python
nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
total_loss = mlm_loss + nsp_loss
```

### 3. **Training Infrastructure**

#### Local Training
```bash
cd /Users/shlba/Desktop/Docs/Study/code/BERT
python src/train_bert.py
```

#### Distributed Training (Multi-GPU)
```bash
torchrun --standalone --nproc_per_node=8 src/train_bert.py
```

#### GPU Training (Remote)
```bash
./scripts/train_gpu_scratch.sh paperspace@184.105.3.177 1
```

### 4. **Training Configuration**
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 6e-4 with learning rate scheduling
- **Batch Size**: Configurable (B=4, T=512 for local training)
- **Mixed Precision**: Uses bfloat16 for GPU efficiency
- **Gradient Accumulation**: For large effective batch sizes

## Evaluation Framework

### 1. **Validation Loss Monitoring**
```python
# Evaluates MLM + NSP loss on validation set every 500 steps
val_loss = model_evaluator.evaluate(val_loader, step, last_step)
```

### 2. **Embedding Quality Assessment**

The model includes sophisticated downstream evaluation using AG News topic classification:

#### AG News Evaluation Process
```python
# Implemented in evaluator.py - evaluate_ag_news_embeddings()
def evaluate_ag_news_embeddings(self, step, max_samples=10000):
```

**Step 1: Data Loading**
- Loads AG News dataset (4-class news topic classification)
- Sample categories: World, Sports, Business, Technology
- Uses 10K training samples, 2.5K test samples for efficiency

**Step 2: Text Encoding**  
```python
# Tokenization with tiktoken encoder
tokenizer = tiktoken.get_encoding("gpt2")
tokens = [CLS_TOKEN] + tokenizer.encode(text) + [SEP_TOKEN]
```

**Step 3: Embedding Extraction**
```python
# Forward pass through BERT encoder
outputs = self.model(input_ids, token_type_ids, attention_mask)
embeddings = outputs['pooler_output']  # [CLS] token representation
```

**Step 4: Classification Training**
```python
# Train logistic regression on BERT embeddings
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(train_embeddings, train_labels)
```

**Step 5: Performance Evaluation**
```python
test_pred = clf.predict(test_embeddings)
test_acc = accuracy_score(test_labels, test_pred)
```

### 3. **Evaluation Schedule**
- **Validation Loss**: Every 500 training steps
- **AG News Embedding Eval**: Every 200 steps (configurable)
- **Checkpoints**: Every 500 steps with model state

### 4. **Monitoring Outputs**
```
Step 1000 AG News Embedding Evaluation:
  Train Accuracy: 0.8543
  Test Accuracy: 0.8121
```

## Key Features

### 1. **Efficient Architecture**
- Compact model (384 hidden, 6 layers) for faster training
- Optimized for Mac/CPU training while supporting GPU
- Memory-efficient attention implementation

### 2. **Robust Training**
- Distributed Data Parallel (DDP) support
- Mixed precision training
- Gradient accumulation for large batch sizes
- Comprehensive checkpointing

### 3. **Evaluation Rigor**
- Downstream task evaluation during training
- Embedding quality assessment via topic classification
- Multiple evaluation metrics and logging

### 4. **Production Ready**
- Docker containerization for GPU training
- GCP integration for data storage
- Automated deployment scripts
- Comprehensive unit tests

## Usage Examples

### Quick Start
```python
from src.training.bert import Bert, BertConfig

# Initialize model
config = BertConfig()
model = Bert(config)

# Forward pass
outputs = model(input_ids, token_type_ids, attention_mask)
embeddings = outputs['pooler_output']  # Sentence embeddings
mlm_predictions = outputs['mlm_logits']  # Token predictions
```

### Extract Text Embeddings
```python
# Use trained model for embedding extraction
model.eval()
with torch.no_grad():
    outputs = model(input_ids, token_type_ids, attention_mask)
    sentence_embeddings = outputs['pooler_output']  # Shape: [batch_size, hidden_size]
```

This implementation provides a complete BERT training and evaluation pipeline with modern MLOps practices and comprehensive monitoring of both training objectives and downstream task performance.
