import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  
import time
import logging
from training.bert_inputs import create_mlm_inputs_and_labels
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, train_loader, config, ddp_config, logger, vocab_size):
        """Initialize trainer with all components"""
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.ddp_config = ddp_config
        self.logger = logger
        self.vocab_size = vocab_size

        assert self.config.total_batch_size % (self.config.B * self.config.T * self.ddp_config['ddp_world_size']) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
        
        self.grad_accum_steps = self.config.total_batch_size // (self.config.B * self.config.T * self.ddp_config['ddp_world_size'])
        
        self.logger_instance = logging.getLogger(__name__)
        
        if self.ddp_config['master_process']:
            self.logger_instance.info(f"total desired batch size: {self.config.total_batch_size}")
            self.logger_instance.info(f"=> calculated gradient accumulation steps: {self.grad_accum_steps}")

    def train_step(self, step, t0):
        """Execute single training step with gradient accumulation"""
        # do one step of the optimization
        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.grad_accum_steps):

            # Get batch
            input_ids, segment_ids, nsp_labels = self.train_loader.next_batch()
            input_ids, segment_ids, nsp_labels = input_ids.to(self.ddp_config['device']), segment_ids.to(self.ddp_config['device']), nsp_labels.to(self.ddp_config['device'])
        
            # Create MLM inputs and labels
            masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)
            masked_input_ids, mlm_labels, attention_mask = masked_input_ids.to(self.ddp_config['device']), mlm_labels.to(self.ddp_config['device']), attention_mask.to(self.ddp_config['device'])
        
            # this field is also used by the forward pass
            if self.ddp_config['ddp']:
                self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
        
            # Forward pass
            with torch.autocast(device_type=self.ddp_config['device'], dtype=torch.bfloat16):
                outputs = self.model(masked_input_ids, segment_ids, attention_mask=attention_mask)
                mlm_scores = outputs['mlm_logits']
                nsp_scores = outputs['nsp_logits']
            
                # MLM loss (only on masked tokens)
                mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.vocab_size), mlm_labels.view(-1), ignore_index=-100)
            
                # NSP loss
                nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)
            
                # Total loss
                total_loss = mlm_loss + nsp_loss
        
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                total_loss = total_loss / self.grad_accum_steps
                loss_accum += total_loss.detach()
                total_loss.backward()

        if self.ddp_config['ddp']:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # determine and set the learning rate for this iteration
        lr = self.config.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()
        
        if self.ddp_config['device_type'] == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_config['ddp_world_size']
        tokens_per_sec = tokens_processed / dt
        
        if self.ddp_config['master_process']:
            self.logger_instance.info(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(self.logger, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")