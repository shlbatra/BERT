
import torch
import os

class CheckpointConfig:
    def __init__(self, checkpoint_dir='checkpoints', file='checkpoint.pt'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_file = os.path.join(self.checkpoint_dir, file)
        with open(self.log_file, "w") as f: # open for writing to clear the file - train loss, val loss
            pass

    @staticmethod
    def save_checkpoint(model, optimizer, config, step, val_loss, log_dir, random_seeds):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': model.config,
            'step': step,
            'val_loss': val_loss.item(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'random_seeds': random_seeds
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)