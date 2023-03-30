import random
import os
from pathlib import Path
import torch
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

import json

import datasets
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    StoppingCriteriaList
)

from codet5_finetune.options import options
from codet5_finetune.distributed import set_distributed_options
from codet5_finetune.util import set_global_seeds
from codet5_finetune.metrics import compute_metrics

from codet5_finetune.data import (
    DataCollatorNTP,
    StoppingCriteriaTokenIds,
    get_debug_pivot_sets,
    assert_debug_data
)


class Trainer42(Seq2SeqTrainer):
    stopping_criteria_list = None
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        if self.stopping_criteria_list is not None:
            gen_kwargs['stopping_criteria'] = self.stopping_criteria_list
        super().evaluate(
            eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs
        )
        

def prepare_model_context(opt):
    ctx = type('Ctx', (), {})()
    ctx.opt = opt

    print(f'{opt.per_device_train_batch_size=}')

    opt.path_java_filtered_subset_root = Path(opt.path_java_filtered_subset_root)

    ctx.ds_data = datasets.load_from_disk(
        opt.path_java_filtered_subset_root / opt.java_filtered_subset_data_dir
    )
    print(f'{len(ctx.ds_data[opt.training_split])=}')
    print(f'{len(ctx.ds_data[opt.eval_split])=}')

    ctx.ds_pivots = datasets.load_from_disk(
        opt.path_java_filtered_subset_root / opt.java_filtered_subset_pivots_dir
    )
    if opt.debug:
        ctx.ds_pivots = get_debug_pivot_sets(ctx.ds_pivots, opt)
        assert_debug_data(ctx, opt)
    else:
        if opt.training_max_samples_count != -1:
            ctx.ds_pivots[opt.training_split] = ctx.ds_pivots[opt.training_split].shuffle(
                seed=opt.seed
            ).select(range(opt.training_max_samples_count))
        if opt.eval_max_samples_count != -1:
            ctx.ds_pivots[opt.eval_split] = ctx.ds_pivots[opt.eval_split].shuffle(
                seed=opt.seed
            ).select(range(opt.eval_max_samples_count))

    print(f'{len(ctx.ds_pivots[opt.training_split])=}')
    print(f'{len(ctx.ds_pivots[opt.eval_split])=}')

    ctx.tokenizer = AutoTokenizer.from_pretrained(opt.base_model_name)

    ctx.data_collator = DataCollatorNTP(
        ctx.ds_data,
        ctx.tokenizer,
        min_encoder_seq_length=opt.min_encoder_seq_length,
        min_decoder_seq_length=opt.min_decoder_seq_length,
        encoder_seq_length=opt.encoder_seq_length,
        decoder_seq_length=opt.decoder_seq_length, 
        append_special_token_to_input=opt.append_special_token_to_input,
    )

    opt.model_dir_base = Path(opt.model_dir_base)
    ctx.model_dir = opt.model_dir_base  / opt.trained_model_name / opt.experiment_name

    # NOTE: move it to compute metric itself?
    examples_dir = ctx.model_dir / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    ctx.examples_dir = examples_dir


    # save steps must be multiple of eval steps and strategy must be the sasme and not no
    # for saving of the best model to work
    assert opt.evaluation_strategy.lower() != 'no'
    assert opt.evaluation_strategy == opt.save_strategy
    assert opt.save_steps % opt.eval_steps == 0

    ctx.args = Seq2SeqTrainingArguments(
        ctx.model_dir,
        evaluation_strategy=opt.evaluation_strategy,
        eval_steps=opt.eval_steps,
        # to enable saving of best model according to docs
        load_best_model_at_end=True, 
        metric_for_best_model=opt.metric_for_best_model,
        greater_is_better=opt.greater_is_better,
        predict_with_generate=opt.predict_with_generate,
        # max generation lenght in eval predictions
        generation_max_length = opt.eval_generate_seq_length, 
        include_inputs_for_metrics=opt.include_inputs_for_metrics,
        logging_strategy=opt.logging_strategy,
        logging_steps=opt.logging_steps,
        save_strategy=opt.save_strategy,
        save_steps=opt.save_steps,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        learning_rate=opt.learning_rate,# if would have been perfect 4e-6 and several epochs
        weight_decay=opt.weight_decay,
        lr_scheduler_type=opt.lr_scheduler_type,
        warmup_steps=opt.warmup_steps,
        save_total_limit=opt.save_total_limit,
        num_train_epochs=opt.num_train_epochs,
        fp16=opt.fp16,
        report_to=opt.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=opt.dataloader_num_workers
    )
    
    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(opt.base_model_name)
    
    def compute_metrics42(preds):
        return compute_metrics(preds, ctx)
    
    ctx.trainer = Trainer42(   
        model_init=model_init,
        args=ctx.args,
        train_dataset=ctx.ds_pivots[opt.training_split],
        eval_dataset=ctx.ds_pivots[opt.eval_split],
        data_collator=ctx.data_collator,
        tokenizer=ctx.tokenizer,
        compute_metrics=compute_metrics42
    )

    if opt.evel_generate_use_eol_stop_tokens:
        ctx.trainer.stopping_criteria_list = StoppingCriteriaList([
            StoppingCriteriaTokenIds(stop_ids=[2, 203, 206], device=opt.device)
        ])

    return ctx

def prepare_options_and_seeds(opt):
    if opt is None:
        opt = options()
        opt = set_distributed_options(opt)
    set_global_seeds(opt)
    return opt

def get_model_context(opt=None):
    opt = prepare_options_and_seeds(opt)
    return prepare_model_context(opt)
