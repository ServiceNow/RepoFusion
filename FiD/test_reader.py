# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import json
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
import multiprocessing
from torch.utils.data import DataLoader, SequentialSampler
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

def evaluate(model, dataset, collator, tokenizer, opt, stopping_criteria, logger, output_path, start_idx=0,
            holes_processed_till_now=[]):
    # sample sequentially for evaluation.
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=min(max(multiprocessing.cpu_count()-1, 1), 20),
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 

    if opt.write_results:
        write_path = output_path / 'test_results'
        fw = open(write_path / ('%d.jsonl'%opt.global_rank), 'a')

    total = 0
    exactmatch = []
    count = 0
    print("start_idx", start_idx)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, labels, _, context_ids, context_mask, hole_id) = batch
            flag = False
            for h_id in hole_id:
                if h_id not in holes_processed_till_now:
                    flag = True
                    break
            # print("context_ids", context_ids.shape)
            # print("context_mask", context_mask.shape)
            # print("labels", labels.shape)
            # print("idx", idx)
            # if i % 100 == 0:
            #     print(i, hole_id, flag)
            if flag:
                #print(i, hole_id, flag)
                if context_ids.size(1) == 0:
                    raise ValueError("The context_ids tensor is empty along dimension 1. Id is {}".format(idx.item()))

                if opt.write_crossattention_scores:
                    model.reset_score_storage()
                
                if opt.model_type == 'codet5':
                    outputs = model.generate(input_ids=context_ids.cuda(),
                                    attention_mask=context_mask.cuda(),
                                    max_length=128)

                if opt.model_type == 'santacoder':
                    outputs = model.generate(input_ids=context_ids.cuda(),
                                    attention_mask=context_mask.cuda(),
                                    max_length=opt.text_maxlength + 128)
                    starting_pos = context_ids.shape[1]
                
                elif opt.model_type == 'codegen':
                    outputs = model.generate(input_ids=context_ids.cuda(),
                                    attention_mask=context_mask.cuda(),
                                    max_length=opt.text_maxlength + 128)
                    starting_pos = context_ids.shape[1]

                if opt.write_crossattention_scores:
                    crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

                for k, o in enumerate(outputs):
                    if opt.model_type == 'codegen' or opt.model_type == 'santacoder':
                        o = o[starting_pos:]
                    ans = tokenizer.decode(o, skip_special_tokens=True)
                    example = dataset.get_example(idx[k].item())
                    gold = example['target']
                    score = src.evaluation.em_code(ans, gold)
                    #print('ans:{}, gold:{}, score:{}'.format(ans, gold, score))
                    exactmatch.append(score)       
                    if opt.write_results:
                        entry = {"id": example['id'], "prediction": ans, "target": example['target']}
                        fw.write(json.dumps(entry) + '\n')
                        fw.flush()
                    if opt.write_crossattention_scores:
                        for j in range(context_ids.size(1)):
                            example['ctxs'][j]['score'] = crossattention_scores[k, j].item()
                    total += 1

                if (i + 1) % opt.eval_print_freq == 0:
                    log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                    if len(exactmatch) == 0:
                        log += '| no answer to compute scores'
                    else:
                        log += f' | average = {np.mean(exactmatch):.3f}'
                    logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return score, total


def run(opt):
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    experiment_name = opt.name + '_' + opt.passage_mode #+ '_' + str(opt.is_append_question)
    output_path = Path(opt.output_dir)/experiment_name
    if opt.is_distributed:
        torch.distributed.barrier()
    output_path.mkdir(parents=True, exist_ok=True)

    if opt.write_results:
        (output_path / 'test_results').mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        output_path / 'run.log'
    )
    
    partial_result_file = os.path.join(output_path, 'test_results', '0.jsonl')
    if os.path.exists(partial_result_file):
        logger.info('Partial result file exists, calculating point of start')
        hole_count = {}
        with open(partial_result_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                entry = json.loads(line)
                hole_id = entry['id']
                if hole_id in hole_count:
                    hole_count[hole_id] += 1
                else:
                    hole_count[hole_id] = 1
            start_idx = len(hole_count)
            holes_processed_till_now = list(hole_count.keys())
    else:
        start_idx = 0
        holes_processed_till_now = []
    logger.info(f'Starting from index {start_idx}')

    model_name = opt.model_name + '-' +opt.model_size
    logger.info(f"Options: {opt}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output Path: {output_path}")

    # Set the tokenizer and stop token IDS (corresponds to EOS, \n and \r in that order).
    if opt.model_type == 'codet5':
        tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
        stop_ids = [2, 203, 206]
    if opt.model_type == 'codegen':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        stop_ids = [50256, 198, 201]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

    if opt.model_type == 'santacoder':
        tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
        stop_ids = [50256, 185, 188]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        
    # Set the model.
    if opt.model_type == 'codet5': 
        # pretrained       
        if opt.trained_model_path is None:
            t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
            model = src.model.FiDT5(t5.config)
            model.load_t5(t5.state_dict())
            logger.info(f"Model initialized from pretrained path: {model_name}")
        else:
            #load_path = Path(opt.trained_model_path) / 'checkpoint' / opt.trained_model_load_type
            load_path = Path(opt.trained_model_path)
            # FiD model
            print(opt.passage_mode)
            if load_path.exists() and not (opt.passage_mode == 'finetuned' or opt.passage_mode == 'toprule+prior'):
                model_class = src.model.FiDT5
                model = model_class.from_pretrained(load_path)
                logger.info(f"Model initialized from FiD path: {load_path}")
            # finetuned model
            else:
                t5 = transformers.T5ForConditionalGeneration.from_pretrained(opt.trained_model_path)
                model = src.model.FiDT5(t5.config)
                model.load_t5(t5.state_dict())
                logger.info(f"Model initialized from finetuned path: {opt.trained_model_path}")

    if opt.model_type == 'codegen':
        print("Starting to load model") 
        print(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, pad_token_id=tokenizer.eos_token_id)
        logger.info(f"Model initialized from {model_name}")

    if opt.model_type == 'santacoder':
        print("Starting to load model") 
        print(opt.model_name)
        model = AutoModelForCausalLM.from_pretrained(opt.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, pad_token_id=tokenizer.eos_token_id)
        logger.info(f"Model initialized from {opt.model_name}")
    model.to(opt.device)

    stopping_criteria = StoppingCriteriaList([src.util.StoppingCriteriaTokenIds(stop_ids=stop_ids, device=opt.device)])

    if opt.passage_mode == 'pretrained' or opt.passage_mode == 'finetuned' or opt.passage_mode == 'toprule+prior':
        is_append_question = False
    else:
        is_append_question = opt.is_append_question
    logger.info(f"Appending question: {is_append_question}")

    collator = src.data.Collator(text_maxlength=opt.text_maxlength, \
                                    tokenizer=tokenizer, \
                                    answer_maxlength=opt.answer_maxlength,
                                    is_append_question=is_append_question,
                                    model_type=opt.model_type,
                                    passage_mode=opt.passage_mode)


    # NOTE: either specify eval_data to load with custom implementation or 
    #       dataset_path to load with hugging face functionality, but not both
    assert (opt.eval_data is not None) or opt.dataset_path is not None


    if opt.dataset_path is None:
        eval_dataset = src.data.Dataset(data_path=opt.eval_data, \
                                        global_rank=opt.global_rank, \
                                        world_size=opt.world_size,\
                                        n_context = opt.n_context, \
                                        tokenizer=tokenizer, \
                                        passage_mode=opt.passage_mode, \
                                        is_append_question=opt.is_append_question, \
                                        text_maxlen=opt.text_maxlength, \
                                        num_of_examples=opt.num_of_eval_examples_per_gpu, \
                                        model_type=opt.model_type,
                                        write_hole_pp_mappings=opt.write_hole_pp_mappings,)
        logger.info(f'Loaded {len(eval_dataset)} validation examples from {opt.eval_data}')
    else:
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
            num_of_examples=opt.num_of_eval_examples_per_gpu,
            model_type=opt.model_type,
            write_hole_pp_mappings=opt.write_hole_pp_mappings,
        )
        logger.info(f'Loaded {len(eval_dataset)} validation examples from {opt.dataset_path}:{opt.eval_split_name}')

    logger.info("Start eval")

    exactmatch, total = evaluate(model, 
                                eval_dataset, 
                                collator, 
                                tokenizer, 
                                opt,
                                stopping_criteria, 
                                logger,
                                output_path,
                                start_idx,
                                holes_processed_till_now)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = output_path / 'test_results'
        write_path = output_path / 'final_output.jsonl'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)
        

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()    
    run(opt)




