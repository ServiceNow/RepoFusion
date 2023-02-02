# Copyright (c) ServiceNow and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This file contains generic utility functionality.
"""

import random
import numpy as np
import os

def set_global_seeds(opt):
    # TODO: add per rank seed mofification
    np.random.seed(opt.seed)
    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    random.seed(opt.seed)
