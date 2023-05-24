"""
This file contains top level entry point for running training or finetuning on models
"""

import os

from Finetuning_CodeT5.model_context import get_model_context


def run(ctx):
    has_checkpoints = any(
        dir.startswith("checkpoint") for dir in os.listdir(ctx.model_dir)
    )
    # Evaluate not trained model
    if not has_checkpoints and ctx.opt.eval_not_trained:
        ctx.trainer.evaluate()
    ctx.trainer.train(resume_from_checkpoint=has_checkpoints)


if __name__ == "__main__":
    run(get_model_context())
