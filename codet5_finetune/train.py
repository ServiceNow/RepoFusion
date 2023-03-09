import random
from pathlib import Path
import torch
import numpy as np

import datasets
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from codet5_finetune.options import options
from codet5_finetune.data import DataCollatorNTP


def prepare(opt):
    ctx = type('Ctx', (), {})()

    print(f'{opt.per_device_train_batch_size=}')

    ctx.ds = datasets.load_from_disk(opt.path_java_filtered_subset)
    print(f'{len(ctx.ds["train"])=}')

    ctx.tokenizer = AutoTokenizer.from_pretrained(opt.base_model_name)

    ctx.data_collator = DataCollatorNTP(
        ctx.tokenizer,
        min_encoder_seq_length=opt.min_encoder_seq_length,
        min_decoder_seq_length=opt.min_decoder_seq_length,
        encoder_seq_length=opt.encoder_seq_length,
        decoder_seq_length=opt.decoder_seq_length, 
        append_special_token_to_input=opt.append_special_token_to_input,
    )

    opt.model_dir_base = Path(opt.model_dir_base)
    model_dir = opt.model_dir_base  / opt.trained_model_name / opt.experiment_name

    ctx.args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy=opt.evaluation_strategy,
        eval_steps=opt.eval_steps,
        predict_with_generate=opt.predict_with_generate,
        include_inputs_for_metrics=opt.include_inputs_for_metrics,
        logging_strategy=opt.logging_strategy,
        logging_steps=opt.logging_steps,
        save_strategy=opt.save_strategy,
        save_steps=opt.save_steps,
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        learning_rate=opt.learning_rate,# if wold have been perfect 4e-6 and several epochs
        weight_decay=opt.weight_decay,
        warmup_steps=opt.warmup_steps,
        save_total_limit=opt.save_total_limit,
        num_train_epochs=opt.num_train_epochs,
        fp16=opt.fp16,
        load_best_model_at_end=False,
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
        return {
            'em': sum(el_pred == el_label for el_pred, el_label in zip(predictions, labels)),
            'example': {'label': labels[0], 'prediction': predictions[0]}
        }
    
    ctx.trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=ctx.args,
        train_dataset=ctx.ds['train'],
        eval_dataset=ctx.ds['test'],
        data_collator=ctx.data_collator,
        tokenizer=ctx.tokenizer,
        compute_metrics=compute_metrics
    )
    
    return ctx
    
    
def run(ctx):
    ctx.trainer.train()

if __name__ == '__main__':
    ctx = prepare(options())
    run(ctx)