"""
This file contains top level entry point for running evaluation on trained checkpoints or default models
"""

import os

from Finetuning_CodeT5.model_context import get_model_context


def run(ctx):
    # we are evaluating stored fine tuned model, not a base one form the hub
    # and on the thole evaluation set
    assert ctx.opt.base_model_name.startswith("/")
    assert ctx.opt.eval_max_samples_count == -1

    print(ctx.trainer.evaluate())


if __name__ == "__main__":
    run(get_model_context())
