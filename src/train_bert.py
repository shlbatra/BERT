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
import sys
import logging

from training.bert import Bert
from training.bert import BertConfig
from data_scripts.dataload import DataLoaderLite
from training.distributed import DDPConfig
from training.config import TrainingConfig
from training.checkpointing import CheckpointConfig
from training.trainer import Trainer
from training.evaluator import Evaluator


if __name__ == "__main__":
    """Train BERT model with MLM and NSP objectives using DDP"""

    # Simple launch:
    # python train_bert.py
    # DDP launch for e.g. 8 GPUs:
    # torchrun --standalone --nproc_per_node=8 train_bert.py

    # Set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

    logger = logging.getLogger(__name__)
     # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ],
        force=True
    )

    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)

    # Capture the seeds for checkpointing
    random_seeds = {
        'torch_seed': 1337,
        'cuda_seed': 1337 if torch.cuda.is_available() else None,
        'random_seed': 1337,
        'numpy_seed': 1337
    }

    # Configurations and setup
    DDPConfig = DDPConfig() # instantiate the config
    distributed_config = DDPConfig.setup_distributed()
    ddp = distributed_config['ddp']
    ddp_rank = distributed_config['ddp_rank']
    ddp_local_rank = distributed_config['ddp_local_rank']
    ddp_world_size = distributed_config['ddp_world_size']
    device = distributed_config['device']
    device_type = distributed_config['device_type']
    logger.info(f"ddp: {ddp}, ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}, device: {device}, device_type: {device_type}")

    TrainingConfig = TrainingConfig() # instantiate the config
    max_steps = TrainingConfig.max_steps
    get_lr = TrainingConfig.get_lr
    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.B # batch size
    T = TrainingConfig.T # sequence length
    weight_decay = TrainingConfig.weight_decay
    starting_learning_rate = TrainingConfig.starting_learning_rate
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)


    # Vocab Size
    VOCAB_SIZE = 50304 # Power of 2 optimization

    # Create model
    model = Bert(BertConfig(vocab_size=VOCAB_SIZE)) 
    model.to(device)

    # Wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Initialize data loaders
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01)

    # create the log directory we will write checkpoints to and log to

    CheckpointConfig = CheckpointConfig() # instantiate the config
    log_file = CheckpointConfig.log_file

    # model trainer and evaluator
    model_trainer = Trainer(raw_model, optimizer, train_loader, TrainingConfig, distributed_config, log_file, VOCAB_SIZE)
    model_evaluator = Evaluator(raw_model, optimizer, TrainingConfig, distributed_config, log_file, VOCAB_SIZE)

    # Training loop
    max_steps = 100000
    eval_interval = 500
    log_interval = 10

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Training step
        model_trainer.train_step(step, t0)
        
        # Once in a while, Run evaluation
        if (step > 0 and step % eval_interval == 0) or last_step:
            val_loss = model_evaluator.evaluate(val_loader, step, last_step)
            if ddp_rank == 0 and val_loss is not None:
                with open(log_file, "a") as f: # open for writing to clear the file - train loss, val loss
                    f.write(f"{step} val {val_loss.item():.4f}\n")
            
            # Run AG News embedding evaluation every 2000 steps
            if step % 2000 == 0 and step > 0:
                ag_news_acc = model_evaluator.evaluate_ag_news_embeddings(step)
                if ddp_rank == 0 and ag_news_acc is not None:
                    logger.info(f"AG News Test Accuracy at step {step}: {ag_news_acc:.4f}")

        # save evaluation and checkpoint every 10000 steps
        if step % 10000 == 0 and step >= 0 and ddp_rank == 0: # 
            # Use a default val_loss if not available
            checkpoint_val_loss = val_loss if 'val_loss' in locals() and val_loss is not None else torch.tensor(0.0)
            CheckpointConfig.save_checkpoint(raw_model, optimizer, BertConfig, step, checkpoint_val_loss, CheckpointConfig.checkpoint_dir, random_seeds=random_seeds)
            logger.info(f"Checkpoint saved at step {step}")

    DDPConfig.destroy_distributed()