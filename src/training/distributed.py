import os
import torch
from torch.distributed import init_process_group, destroy_process_group

class DDPConfig:

    @classmethod
    def setup_distributed(cls):
        """Setup distributed training environment"""
        # set up DDP (distributed data parallel).
        # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)

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
            return {
                'ddp': ddp,
                'ddp_rank': ddp_rank,
                'ddp_local_rank': ddp_local_rank,
                'ddp_world_size': ddp_world_size,
                'master_process': master_process,
                'device': device,
                'device_type': device_type
            }
        else:
            # vanilla, non-DDP run
            ddp_rank = 0
            ddp_local_rank = 0
            ddp_world_size = 1
            master_process = True
            return {
                'ddp': ddp,
                'ddp_rank': ddp_rank,
                'ddp_local_rank': ddp_local_rank,
                'ddp_world_size': ddp_world_size,
                'master_process': master_process,
                'device': device,
                'device_type': device_type
            }
    
    @classmethod
    def destroy_distributed(cls):
        if cls.ddp:
            destroy_process_group()
