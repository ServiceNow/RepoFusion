import random
import numpy as np
import os

import torch
from transformers import StoppingCriteria 

def set_global_seeds(opt):
    # TODO: add per rank seed mofification
    #       and add rank setting in options
    np.random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    random.seed(opt.seed)

class StoppingCriteriaTokenIds(StoppingCriteria):
    def __init__(self, stop_ids, device='cuda'):
        super().__init__()
        self.stop_ids = torch.tensor(stop_ids).to(device)

    def __call__(self, input_ids, scores, *kwargs):
        # true if any alement is one of the stop ids in all rows
        return torch.all(torch.any(
            torch.isin(input_ids, self.stop_ids),
            dim=1,
            keepdim=False
        )).item()
