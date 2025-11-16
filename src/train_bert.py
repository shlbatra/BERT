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

from bert import Bert
from bert_inputs import create_mlm_inputs_and_labels
from data_scripts.dataload import DataLoaderLite


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
    'max_position_embeddings': max_seq_len  # 'intermediate_size': 3072,
}

# Create model
model = Bert(**model_config)
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