import random
from pathlib import Path
import torch

import datasets
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from codet5_finetune.options import options
from codet5_finetune.data import DataCollatorNTP

    
def run(opt):
    print(f'{opt.per_device_train_batch_size=}')

    ds = datasets.load_from_disk(opt.path_java_filtered_subset)
    print(f'{len(ds["train"])=}')

    tokenizer = AutoTokenizer.from_pretrained(opt.base_model_name)

    data_collator = DataCollatorNTP(
        tokenizer,
        min_encoder_seq_length=opt.min_encoder_seq_length,
        min_decoder_seq_length=opt.min_decoder_seq_length,
        encoder_seq_length=opt.encoder_seq_length,
        decoder_seq_length=opt.decoder_seq_length
    )

    opt.model_dir_base = Path(opt.model_dir_base)
    model_dir = opt.model_dir_base  / opt.trained_model_name / opt.experiment_name

    args = Seq2SeqTrainingArguments(
        model_dir,
        logging_strategy=opt.logging_strategy,
        logging_steps=opt.logging_steps,
        save_strategy=opt.save_strategy,
        save_steps=opt.save_steps,
        learning_rate=opt.learning_rate,# if wold have been perfect 4e-6 and several epochs
        per_device_train_batch_size=opt.per_device_train_batch_size,
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        weight_decay=opt.weight_decay,
        save_total_limit=opt.save_total_limit,
        num_train_epochs=opt.num_train_epochs,
        fp16=opt.fp16,
        load_best_model_at_end=False,
        report_to=opt.report_to,
        remove_unused_columns=False
    )
    
    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(opt.base_model_name)

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=None
    )
    
    trainer.train()

if __name__ == '__main__':
    run(options())