import random
from pathlib import Path
import torch
import numpy as np

import json

import datasets
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from codet5_finetune.options import options
from codet5_finetune.util import set_global_seeds
from codet5_finetune.data import DataCollatorNTP, get_debug_pivot_sets, assert_debug_data

step = 0

def prepare(opt):
    ctx = type('Ctx', (), {})()

    print(f'{opt.per_device_train_batch_size=}')

    ctx.ds_data = datasets.load_from_disk(opt.path_java_filtered_subset)
    print(f'{len(ctx.ds_data["train"])=}')
    print(f'{len(ctx.ds_data["validation"])=}')

    ctx.ds_pivots = datasets.load_from_disk(opt.path_java_filtered_subset_pivots)
    if opt.debug:
        ctx.ds_pivots = get_debug_pivot_sets(ctx.ds_pivots, opt)
        assert_debug_data(ctx, opt)
    print(f'{len(ctx.ds_pivots["train"])=}')
    print(f'{len(ctx.ds_pivots["validation"])=}')

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
    model_dir = opt.model_dir_base  / opt.trained_model_name / opt.experiment_name

    examples_dir = model_dir / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)


    # save steps must be multiple of eval steps and strategy must be the sasme and not no
    # for saving of the best model to work
    assert opt.evaluation_strategy.lower() != 'no'
    assert opt.evaluation_strategy == opt.save_strategy
    assert opt.save_steps % opt.eval_steps == 0

    ctx.args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy=opt.evaluation_strategy,
        eval_steps=opt.eval_steps,
        load_best_model_at_end=True, # to enable saving of best model according to docs
        metric_for_best_model=opt.metric_for_best_model,
        greater_is_better=opt.greater_is_better,
        predict_with_generate=opt.predict_with_generate,
        include_inputs_for_metrics=opt.include_inputs_for_metrics,
        logging_strategy=opt.logging_strategy,
        logging_steps=opt.logging_steps,
        save_strategy=opt.save_strategy,
        save_steps=opt.save_steps,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        learning_rate=opt.learning_rate,# if would have been perfect 4e-6 and several epochs
        weight_decay=opt.weight_decay,
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
        labels_first_line = [el.splitlines()[0] for el in labels]
        predictions_first_line = [el.splitlines()[0] for el in predictions]

        # TODO: get several exactly matched and several not matched samples instead
        examples = {
            'full': [
                {'input': input, 'label': label, 'prediction': prediction}
                for input, label, prediction in zip(inputs, labels, predictions)
            ],
            'first_line': [
                 {'input': input, 'label': label, 'prediction': prediction}
                for input, label, prediction in zip(inputs, labels_first_line, predictions_first_line)
            ]
        }
        global step
        example_file = examples_dir / f'{step}.json'
        with example_file.open('wt') as f:
            json.dump(examples, f)
        step += 1
        def count_exact_matches(a, b):
            return sum(el_a == el_b for el_a, el_b in zip(a, b))

        return {
            'em': count_exact_matches(predictions, labels),
            'em_first_line': count_exact_matches(labels_first_line, predictions_first_line),
            #'examples': str(examples)
        }
    
    ctx.trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=ctx.args,
        train_dataset=ctx.ds_pivots['train'],
        eval_dataset=ctx.ds_pivots['validation'],
        data_collator=ctx.data_collator,
        tokenizer=ctx.tokenizer,
        compute_metrics=compute_metrics
    )
    
    return ctx
    
    
def run(ctx):
    ctx.trainer.train()

if __name__ == '__main__':
    opt = options()
    set_global_seeds(opt)
    ctx = prepare(opt)
    run(ctx)