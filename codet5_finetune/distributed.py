import torch
import os

def set_distributed_options(opt):
    # adapted from FiD distributed options setting
    # multi gpu options
    if 'LOCAL_RANK' in os.environ:
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        has_local_rank = True
    else:
        has_local_rank = False
    
    if has_local_rank and opt.local_rank != -1:
        opt.global_rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.n_gpu_per_node = int(os.environ['LOCAL_WORLD_SIZE'])

        # number of nodes / node ID
        opt.n_nodes = opt.world_size // opt.n_gpu_per_node
        opt.node_id = opt.global_rank // opt.n_gpu_per_node
        opt.is_distributed = True
    else:
        n_gpu = torch.cuda.device_count()
        opt.n_nodes = 1
        opt.node_id = 0
        opt.local_rank = 0
        opt.global_rank = 0
        opt.world_size = n_gpu
        opt.n_gpu_per_node = n_gpu
        opt.is_distributed = False

    # define whether this is the master process / if we are in distributed mode
    opt.is_main = opt.node_id == 0 and opt.local_rank == 0
    opt.multi_node = opt.n_nodes > 1
    opt.multi_gpu = opt.world_size > 1

    # set GPU device
    if opt.is_distributed:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device

    return opt
