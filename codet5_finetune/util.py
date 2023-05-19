import random
import numpy as np
import os


def set_global_seeds(opt):
    # TODO: add per rank seed mofification
    np.random.seed(opt.seed)
    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    random.seed(opt.seed)
