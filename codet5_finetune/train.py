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
from codet5_finetune.util import (
    set_global_seeds,
    StoppingCriteriaTokenIds,
    set_distributed_options,
)
from codet5_finetune.data import DataCollatorNTP, get_debug_pivot_sets, assert_debug_data

step = 0

def exact_match_a2p_ratio(p, l):
    cnt = 0
    for ip, il in zip(p, l):
        if ip == il:
            cnt += 1
        else:
            break
    l  = max(len(l), len(p))
    # if empty label and prediction,- match
    if cnt == 0 and l == 0:
        return 1.0
    r =  cnt / l
    return r

def average_exact_match_a2p_ratio(predictions, labels):
    return sum(
        exact_match_a2p_ratio(p, l)
        for p, l in zip(predictions, labels)
    ) / len(labels)

def exact_matches_ratio(a, b):
    return sum(el_a == el_b for el_a, el_b in zip(a, b)) / len(a)


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
        

def prepare(opt):
    ctx = type('Ctx', (), {})()

    print(f'{opt.per_device_train_batch_size=}')

    ctx.ds_data = datasets.load_from_disk(opt.path_java_filtered_subset)
    print(f'{len(ctx.ds_data[opt.training_split])=}')
    print(f'{len(ctx.ds_data[opt.eval_split])=}')

    ctx.ds_pivots = datasets.load_from_disk(opt.path_java_filtered_subset_pivots)
    if opt.debug:
        ctx.ds_pivots = get_debug_pivot_sets(ctx.ds_pivots, opt)
        assert_debug_data(ctx, opt)
    else:
        # cap validation for 10K for now for speed
        ctx.ds_pivots[opt.eval_split] = ctx.ds_pivots[opt.eval_split].shuffle(seed=opt.seed).select(range(10000))

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

    examples_dir =ctx.model_dir / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)


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
    
    def compute_metrics(preds):
        label_ids = np.where(preds.label_ids != -100, preds.label_ids, ctx.tokenizer.pad_token_id)
        labels = ctx.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = ctx.tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
        inputs = ctx.tokenizer.batch_decode(preds.inputs)
        # if opt.strip_new_lines_for_em:
        #     def strip_get_first_line(items):
        #         items = [el.splitlines() for el in items]
        #         items_first_line = [el[0] for el in items]
        #         items = [''.join(el) for el in items]
        #         return items, items_first_line
        #     labels, labels_first_line = strip_get_first_line(labels)
        #     predictions, predictions_first_line = strip_get_first_line(predictions)
        # else:
        def get_first_lines(vals):
            return [
                lines[0] if len(lines) > 0 else ''
                for lines in (
                    el.splitlines()
                    for el in vals
                )
            ]
        labels_first_line = get_first_lines(labels)
        predictions_first_line = get_first_lines(predictions)

        # TODO: get several exactly matched and several not matched samples instead
        # for now save just 500 of first examples
        sz = min(500, len(inputs))
        examples = {
            'full': [
                {'input': input, 'label': label, 'prediction': prediction}
                for input, label, prediction in zip(inputs[:sz], labels[:sz], predictions[:sz])
            ],
            'first_line': [
                 {'input': input, 'label': label, 'prediction': prediction}
                for input, label, prediction in zip(inputs[:sz], labels_first_line[:sz], predictions_first_line[:sz])
            ]
        }
        # TODO: get step from trainer or create run specific prefix
        #       now overwrites on restart
        global step
        if opt.is_main:
            # in case the fuction is called on all processes, will work on one node only
            pid = os.getpid()
            example_file = examples_dir / f'{step}_{pid}.json'
            with example_file.open('wt') as f:
                json.dump(examples, f)
            step += 1

        return {
            'em_ratio': exact_matches_ratio(predictions, labels),
            'em_a2p_ratio': average_exact_match_a2p_ratio(predictions, labels),
            'em_first_line_ratio': exact_matches_ratio(labels_first_line, predictions_first_line),
            #'examples': str(examples)
        }
    
    ctx.trainer = Trainer42(   
        model_init=model_init,
        args=ctx.args,
        train_dataset=ctx.ds_pivots[opt.training_split],
        eval_dataset=ctx.ds_pivots[opt.eval_split],
        data_collator=ctx.data_collator,
        tokenizer=ctx.tokenizer,
        compute_metrics=compute_metrics
    )

    if opt.evel_generate_use_eol_stop_tokens:
        ctx.trainer.stopping_criteria_list = StoppingCriteriaList([
            StoppingCriteriaTokenIds(stop_ids=[2, 203, 206], device=opt.device)
        ])

    return ctx
    
    
def run(ctx):
    ctx.trainer.train(
        resume_from_checkpoint=any(
            dir.startswith("checkpoint") for dir in os.listdir(ctx.model_dir)
        )
    )

if __name__ == '__main__':
    opt = options()
    opt = set_distributed_options(opt)
    set_global_seeds(opt)
    ctx = prepare(opt)
    run(ctx)