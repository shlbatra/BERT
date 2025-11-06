import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import time
import math
import random
import numpy as np

from bert_model import BertForPreTraining
from data_scripts.dataload import DataLoaderLite


def create_mlm_inputs_and_labels(input_ids, mask_token_id=50258, pad_token_id=50259, mask_prob=0.15):
    """
    Create masked language model inputs and labels. BERT randomly assigns masks to 15% of sequence. 
    In this 15%, 80% are replaced with [MASK], 10% are replaced with random token, and 10% are left unchanged.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        mask_token_id: Token ID for [MASK]
        pad_token_id: Token ID for [PAD] 
        mask_prob: Probability of masking each token
    
    Returns:
        masked_input_ids: Input with some tokens masked
        mlm_labels: Labels for MLM task (-100 for non-masked tokens)
    """
    masked_input_ids = input_ids.clone()
    mlm_labels = input_ids.clone()
    
    # Create probability matrix for masking based on mask_prob
    probability_matrix = torch.full(input_ids.shape, mask_prob) # 15% of tokens to be masked
    
    # Don't mask special tokens ([CLS], [SEP], [PAD])
    special_tokens_mask = (
        (input_ids == 50256) |  # [CLS]
        (input_ids == 50257) |  # [SEP]  
        (input_ids == pad_token_id)  # [PAD]
    )
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Create mask
    masked_indices = torch.bernoulli(probability_matrix).bool() # 80% masked
    
    # Only compute loss on masked tokens
    mlm_labels[~masked_indices] = -100 # loss ignored for non masked tokens
    
    # 80% of masked tokens become [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices # Out of 15% masked, 80% to [MASK]
    masked_input_ids[indices_replaced] = mask_token_id # replace mask index with [MASK] token id
    
    # 10% of masked tokens stay the same and rest 10% are assigned a random word (but we still predict them)
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced # Out of  15% masked, 10%/20% to random
    random_words = torch.randint(50260, input_ids.shape, dtype=torch.long)
    masked_input_ids[indices_random] = random_words[indices_random]
    
    return masked_input_ids, mlm_labels


def get_lr(step, warmup_steps=10000, max_lr=6e-4, max_steps=100000):
    """Learning rate schedule with warmup and cosine decay"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return 0.0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff


# Simple launch:
# python train_bert.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_bert.py

# Set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)

# Model and training hyperparameters
batch_size = 8
max_seq_len = 512
vocab_size = 50260

# BERT config (BERT-Base)
model_config = {
    'vocab_size': vocab_size,
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12, 
    'intermediate_size': 3072,
    'max_position_embeddings': max_seq_len
}

# Create model
model = BertForPreTraining(**model_config)
model.to(device)

# Wrap model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Initialize data loaders
train_loader = DataLoaderLite(B=batch_size, T=max_seq_len, process_rank=ddp_rank, 
                             num_processes=ddp_world_size, split="train", master_process=master_process)
val_loader = DataLoaderLite(B=batch_size, T=max_seq_len, process_rank=ddp_rank,
                           num_processes=ddp_world_size, split="val", master_process=master_process)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01)

# Training loop
max_steps = 100000
eval_interval = 500
log_interval = 10

for step in range(max_steps):
    t0 = time.time()
    
    # Set learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Get batch
    input_ids, segment_ids, nsp_labels = train_loader.next_batch()
    input_ids, segment_ids, nsp_labels = input_ids.to(device), segment_ids.to(device), nsp_labels.to(device)
    
    # Create MLM inputs and labels
    masked_input_ids, mlm_labels = create_mlm_inputs_and_labels(input_ids)
    masked_input_ids, mlm_labels = masked_input_ids.to(device), mlm_labels.to(device)
    
    # Forward pass
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        mlm_scores, nsp_scores = model(masked_input_ids, segment_ids)
        
        # MLM loss (only on masked tokens)
        mlm_loss = F.cross_entropy(mlm_scores.view(-1, vocab_size), mlm_labels.view(-1), ignore_index=-100)
        
        # NSP loss
        nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)
        
        # Total loss
        total_loss = mlm_loss + nsp_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = batch_size * max_seq_len * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    
    if master_process and step % log_interval == 0:
        print(f"step {step:5d} | loss: {total_loss.item():.6f} | mlm: {mlm_loss.item():.6f} | nsp: {nsp_loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    # Evaluation
    if step > 0 and step % eval_interval == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                input_ids, segment_ids, nsp_labels = val_loader.next_batch()
                input_ids, segment_ids, nsp_labels = input_ids.to(device), segment_ids.to(device), nsp_labels.to(device)
                
                masked_input_ids, mlm_labels = create_mlm_inputs_and_labels(input_ids)
                masked_input_ids, mlm_labels = masked_input_ids.to(device), mlm_labels.to(device)
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    mlm_scores, nsp_scores = model(masked_input_ids, segment_ids)
                    mlm_loss = F.cross_entropy(mlm_scores.view(-1, vocab_size), mlm_labels.view(-1), ignore_index=-100)
                    nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)
                    loss = mlm_loss + nsp_loss
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item()/val_loss_steps:.4f}")
        model.train()

if ddp:
    destroy_process_group()