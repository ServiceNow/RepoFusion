# Copyright (c) ServiceNow and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This file contains config options parsing:
"""

import yaml
import sys
from pathlib import Path


def options(args=None):
    """
    A placeholder for options to be replaced with actual arg parsing later
      meant to be run from a pwd of the repo root or a path to the common
      config specified in args['base']
    """
    base_config = "Finetuning_CodeT5/config/base.yaml"
    experiment_config = None
    if args is not None:
        if isinstance(args, dict):
            if "base_config" in args:
                base_config = args["base_config"]
            if "experiment_config" in args:
                experiment_config = args["experiment_config"]
        else:
            experiment_config = args
    elif len(sys.argv) > 1:
        experiment_config = sys.argv[1]

    base_config = Path(base_config)
    if experiment_config is not None:
        experiment_config = Path(experiment_config)

    print(sys.argv)

    with open(base_config) as f:
        data = yaml.safe_load(f)
        data["experiment_name"] = base_config.stem

    if experiment_config is not None:
        with open(experiment_config) as f:
            data.update(yaml.safe_load(f))
        data["experiment_name"] = experiment_config.stem

    def opt_repr(self):
        return f"Opt() at {id(self):x}:\n" + "".join(
            f"  {el}: {getattr(self, el)}\n"
            for el in dir(self)
            if not el.startswith("__")
        )

    data["__repr__"] = opt_repr
    opt = type("Opt", (), data)()
    return opt
