import torch
import torch.distributed as dist
import torch.nn.functional as F
import logging
from training.bert_inputs import create_mlm_inputs_and_labels

class Evaluator:
    def __init__(self, model, optimizer, config, ddp_config, logger, vocab_size):
        """Initialize trainer with all components"""
        self.model = model
        self.model.eval()
        self.optimizer = optimizer
        self.config = config
        self.ddp_config = ddp_config
        self.logger = logger
        self.vocab_size = vocab_size

        self.logger_instance = logging.getLogger(__name__)
        
        assert self.config.total_batch_size % (self.config.B * self.config.T * self.ddp_config['ddp_world_size']) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
        grad_accum_steps = self.config.total_batch_size // (self.config.B * self.config.T * self.ddp_config['ddp_world_size'])
        if self.ddp_config['master_process']:
            self.logger_instance.info(f"total desired batch size: {self.config.total_batch_size}")
            self.logger_instance.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    def evaluate(self, val_loader, step, last_step):
        """Execute evaluate step"""
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):

                input_ids, segment_ids, nsp_labels = val_loader.next_batch()
                input_ids, segment_ids, nsp_labels = input_ids.to(self.ddp_config['device']), segment_ids.to(self.ddp_config['device']), nsp_labels.to(self.ddp_config['device'])
                masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)

                x, y = val_loader.next_batch()
                x, y = x.to(self.ddp_config['device']), y.to(self.ddp_config['device'])
                with torch.autocast(device_type=self.ddp_config['device_type'], dtype=torch.bfloat16):
                    outputs = self.model(masked_input_ids, segment_ids, attention_mask=attention_mask)
                    nsp_scores = outputs['nsp_logits']
                    mlm_scores = outputs['mlm_logits']
                    mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.vocab_size), mlm_labels.view(-1), ignore_index=-100)
                    nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)
                    loss = mlm_loss + nsp_loss
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

        if self.ddp_config['ddp']:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.ddp_config['master_process']:
            self.logger_instance.info(f"validation loss: {val_loss_accum.item():.4f}")
            return val_loss_accum
