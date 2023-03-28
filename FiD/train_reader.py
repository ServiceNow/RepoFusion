
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

import time
import sys
import torch
import transformers#
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
import multiprocessing
from transformers import StoppingCriteriaList

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model


def generate_and_calculate_em(model, tokenizer, dataset, idx, context_ids, context_mask, stopping_criteria):
    model = model.module if hasattr(model, "module") else model
    outputs = model.generate(input_ids=context_ids.cuda(),
                            attention_mask=context_mask.cuda(),
                            max_length=100)
                            # stopping_criteria=stopping_criteria)
    total = 0
    exactmatch = []
    for k, o in enumerate(outputs):
        ans = tokenizer.decode(o, skip_special_tokens=True)
        gold = dataset.get_example(idx[k])['target']
        score = src.evaluation.em_code(ans, gold)
        #print('ans:{}, gold:{}, score:{}'.format(ans, gold, score))
        total += 1
        exactmatch.append(score)

    return exactmatch, total, (ans, gold)

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path, tokenizer, logger, stopping_criteria):
    global torch
    if opt.is_main:
        try:
            import torch.utils.tensorboard
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(checkpoint_path))
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    # sample random holes for training.
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=min(max(multiprocessing.cpu_count()-1, 1), 10),
        collate_fn=collator
    )

    curr_loss = 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            # if step >= 235:
            #     print("step", step)
            if step % 100 == 0:
                logger.info(f"Step {step} / {opt.total_steps}")

            (idx, labels, _, context_ids, context_mask) = batch
            # print("context_ids", context_ids.shape)
            # print("context_mask", context_mask.shape)
            # print("labels", labels.shape)
            # print("idx", idx)

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                return_dict=False
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            # the training loss is calculated over the whole batch.
            curr_loss += train_loss.item()

            if step % opt.eval_loss_freq == 0:
                # the validation loss and EM are calculted over the whole validation dataset.
                logger.info(f"Evaluating validation set loss")
                dev_loss = evaluate_loss(model, eval_dataset, collator, opt)
                model.train()
                if opt.is_main:
                    log = f"{step} / {opt.total_steps} |"
                    log += f"training loss: {curr_loss/opt.eval_loss_freq:.3f} |"
                    log += f"validation loss: {dev_loss} |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation Loss", dev_loss, step)
                        tb_logger.add_scalar("Training Loss", curr_loss / (opt.eval_loss_freq), step)
                    curr_loss = 0.

            if step % opt.eval_em_freq == 0:
                dev_em = evaluate_em(model, eval_dataset, tokenizer, collator, opt, stopping_criteria)
                # calculate train EM only over the current batch to save computation.
                exactmatch, total, (sample_ans, sample_gold) = generate_and_calculate_em(model, tokenizer, \
                                                                                        train_dataset, idx, \
                                                                                        context_ids, context_mask,\
                                                                                        stopping_criteria)
                sample_train_em, _ = src.util.weighted_average(np.mean(exactmatch), total, opt)
                model.train()

                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                    opt, checkpoint_path, 'best_dev')
                    logger.info(f"Saving best dev EM checkpoint to {checkpoint_path}")
                    log = f"{step} / {opt.total_steps} |"
                    log += f"sample training EM: {100*sample_train_em:.2f}EM |"
                    log += f"evaluation EM: {100*dev_em:.2f}EM |"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation EM", dev_em, step)
                        tb_logger.add_scalar("Training EM", sample_train_em, step)
                        tb_logger.add_text("Training Sample Predictions", \
                                        str({'pred': sample_ans, 'gold': sample_gold}), step)

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                        opt, checkpoint_path, f"step-{step}")
                logger.info(f"Saving checkpoint to {checkpoint_path}")

            if step > opt.total_steps:
                break
    logger.info(f"Training finished after {epoch} epochs and {step} steps.")

def evaluate_loss(model, dataset, collator, opt):
    # sample sequntially for evaluation.
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=min(max(multiprocessing.cpu_count()-1, 1), 10),
        collate_fn=collator
    )
    model.eval()
    step = 0
    curr_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            eval_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                return_dict=False
            )[0]

            eval_loss = src.util.average_main(eval_loss, opt)
            #print(eval_loss)
            curr_loss += eval_loss.item() 
    return curr_loss / step
    
def evaluate_em(model, dataset, tokenizer, collator, opt, stopping_criteria):
    # sample sequntially for evaluation.
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=min(max(multiprocessing.cpu_count()-1, 1), 10),
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, labels, _, context_ids, context_mask) = batch
            em, count, _ = generate_and_calculate_em(model, tokenizer, dataset, \
                                                    idx, context_ids, context_mask, stopping_criteria)
            #print("EM:{em}, count:{count}, step: {step}".format(em=em, count=count, step=step))
            exactmatch.extend(em)
            total += count
            #print("total:{total}".format(total=total))

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

def run(opt):
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    experiment_name = opt.name + '_' + opt.passage_mode + '_' + str(opt.is_append_question)
    checkpoint_path = Path(opt.checkpoint_dir)/experiment_name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    # checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = opt.model_name + '-' +opt.model_size
    model_class = src.model.FiDT5
    logger.info(f"Options: {opt}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Checkpoint Path: {checkpoint_path}")

    # Get max model lenght
    model_cfg = transformers.AutoConfig.from_pretrained(model_name)
    if not (hasattr(model_cfg, 'n_positions') and hasattr(model_cfg, 'output_past')):
        raise ValueError(f'Model {model_name} config has no n_positions and output_past')

    model_max_length = opt.model_max_length
    # Set it from opt?
    output_past = bool(model_cfg.output_past)

    if output_past is False and model_max_length > model_cfg.n_positions:
        raise ValueError(f'max_model_length is bigger than n_positions for output_past == False')
    
    # Set the tokenizer and initialize the collator.
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    stopping_criteria = StoppingCriteriaList([src.util.StoppingCriteriaTokenIds(stop_ids=[2, 203, 206], device=opt.device)])

    collator = src.data.Collator(text_maxlength=opt.text_maxlength, \
                                    tokenizer=tokenizer, \
                                    answer_maxlength=opt.answer_maxlength,
                                    is_append_question=opt.is_append_question)

    # Load the training and validation data. The data is split across multiple GPUs.

    # NOTE: either specify train_data and eval_data to load with custom implementation or 
    #       dataset_path to load with hugging face functionality, but not both
    assert (opt.train_data is not None and opt.eval_data is not None) or opt.dataset_path is not None


    if opt.dataset_path is None:
        train_dataset = src.data.Dataset(data_path=opt.train_data, \
                                        global_rank=opt.global_rank, \
                                        world_size=opt.world_size,\
                                        n_context = opt.n_context, \
                                        tokenizer=tokenizer, \
                                        passage_mode=opt.passage_mode, \
                                        is_append_question=opt.is_append_question, \
                                        text_maxlen=opt.text_maxlength)
        logger.info(f'Loaded {len(train_dataset)} training examples from {opt.train_data}')

        eval_dataset = src.data.Dataset(data_path=opt.eval_data, \
                                        global_rank=opt.global_rank, \
                                        world_size=opt.world_size,\
                                        n_context = opt.n_context, \
                                        tokenizer=tokenizer, \
                                        passage_mode=opt.passage_mode, \
                                        is_append_question=opt.is_append_question, \
                                        text_maxlen=opt.text_maxlength, \
                                        num_of_examples=opt.num_of_eval_examples_per_gpu)
        logger.info(f'Loaded {len(eval_dataset)} validation examples from {opt.eval_data}')
    else:
        train_dataset = src.data.Dataset(
            data_path=opt.dataset_path,
            features_format_file=opt.features_format_file,
            data_file_pattern=opt.data_file_pattern,
            split=opt.train_split_name,
            hf_datasets_cache_dir=opt.hf_datasets_cache_dir,
            hf_datasets_load_num_proc=opt.hf_datasets_load_num_proc,
            global_rank=opt.global_rank,
            world_size=opt.world_size,
            n_context = opt.n_context,
            tokenizer=tokenizer,
            passage_mode=opt.passage_mode,
            is_append_question=opt.is_append_question,
            text_maxlen=opt.text_maxlength
        )
        logger.info(f'Loaded {len(train_dataset)} training examples from {opt.dataset_path}:{opt.train_split_name}')

        eval_dataset = src.data.Dataset(
            data_path=opt.dataset_path,
            features_format_file=opt.features_format_file,
            data_file_pattern=opt.data_file_pattern,
            split=opt.eval_split_name,
            hf_datasets_cache_dir=opt.hf_datasets_cache_dir,
            hf_datasets_load_num_proc=opt.hf_datasets_load_num_proc,
            global_rank=opt.global_rank,
            world_size=opt.world_size,
            n_context = opt.n_context,
            tokenizer=tokenizer,
            passage_mode=opt.passage_mode,
            is_append_question=opt.is_append_question,
            text_maxlen=opt.text_maxlength,
            num_of_examples=opt.num_of_eval_examples_per_gpu
        )
        logger.info(f'Loaded {len(eval_dataset)} validation examples from {opt.dataset_path}:{opt.eval_split_name}')


    load_path = checkpoint_path / 'checkpoint' / 'latest' 
    # if checkpoint does not exist or if checkpoint exists but load path does not exist, train from scratch. The latter might happen when the 
    # training got interrupted before the checkpoint could be stored.
    if (not checkpoint_exists and opt.model_path == "none") or (checkpoint_exists and not load_path.exists()):
        logger.info(f'Training model from scratch')
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    # if checkpoint exists and load path exists, load the model from the checkpoint.
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'    
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
        logger.info(f"Resuming training from step {step}")
        logger.info(f"Best dev EM: {best_dev_em:.2f}")
    # if model path is specified, load the model from the model path (can be different from the usual checkpoint path).
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")
        logger.info(f"Resuming training from step {step}")
        logger.info(f"Best dev EM: {best_dev_em:.2f}")
        logger.info(f"Optimizer and scheduler are being reset.")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    
    logger.info("Start training")
    # the training and evaluation loop.

    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path,
        tokenizer,
        logger,
        stopping_criteria,
    )

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)
    
    run(opt)
    
