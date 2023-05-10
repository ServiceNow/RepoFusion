import openai
import time
import os
import torch
import json
import numpy as np
from pathlib import Path
import torch.distributed as dist
import multiprocessing
from torch.utils.data import DataLoader, SequentialSampler
from transformers import GPT2TokenizerFast, AutoTokenizer
from text_generation import Client
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

def generate_prediction(prompt):
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",\
                                            messages = [
                                                {"role": "system", "content": "In the following conversation, whenever you are given an incomplete snippet of Java code, you will complete it until the end of the line. I only want a single line of output."},
                                                {"role": "user", "content": prompt},
                                            ], \
                                            max_tokens=128,\
                                            #stop=["\n"],\
                                            temperature=0.0)
    except:
        print ("Waiting")
        response = None
    return response

def evaluate(dataset, tokenizer, opt, logger, output_path, start_idx=0, client=None):
    # sample sequentially for evaluation.
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=min(max(multiprocessing.cpu_count()-1, 1), 20),
    )
 
    if opt.write_results:
        write_path = output_path / 'test_results'
        fw = open(write_path / ('%d.jsonl'%opt.global_rank), 'a')

    total = 0
    exactmatch = []
    count = 0
    for i, batch in enumerate(dataloader):
        idx, gold, hole_id, prompt = batch
        gold = gold[0]
        hole_id = hole_id[0]
        prompt = prompt[0]
        #print('gold:{}, hole_id:{}, prompt:{}'.format(gold, hole_id, prompt))
        count += (i + 1) * opt.per_gpu_batch_size
        if count < start_idx:
            continue

        if opt.model_type == 'chatgpt':
            response = generate_prediction(prompt)
            while(response is None):
                time.sleep(60)
                logger.info('Waiting for response for hole_id:{}'.format(hole_id))
                response = generate_prediction(prompt)
            if response is not None:
                ans = response.choices[0].message.content

        if opt.model_type == 'starcoder':
            if prompt:
                try:
                    ans = client.generate(prompt, max_new_tokens=128).generated_text
                except:
                    import subprocess
                    # Execute the command and capture its output
                    result = subprocess.run(['eai', 'login', 'token'], stdout=subprocess.PIPE)
                    # Get the output as a string
                    access_token = result.stdout.decode('utf-8').strip()
                    client = Client("https://snow-llmd-snow_dzmitry_llmd_bigcode_job.job.console.elementai.com", \
                                    headers={"Authorization": "Bearer " + access_token})
                    ans = client.generate(prompt, max_new_tokens=128).generated_text


        score = src.evaluation.em_code(ans, gold)
        #print('ans:{}, gold:{}, score:{}'.format(ans, gold, score))
        exactmatch.append(score)       
        if opt.write_results:
            entry = {"id": hole_id, "prediction": ans, "target": gold}
            fw.write(json.dumps(entry) + '\n')
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

    experiment_name = opt.name 
    output_path = Path(opt.output_dir)/experiment_name
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
        with open(partial_result_file, 'r') as f:
            lines = f.readlines()
            start_idx = len(lines)
    else:
        start_idx = 0
    logger.info(f'Starting from index {start_idx}')

    if opt.model_type == 'starcoder':
        hf_access_token = "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"
        tokenizer = AutoTokenizer.from_pretrained("bigcode/large-model", use_auth_token=hf_access_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        # ACCESS_KEY = "yhAb0IDJytoBLCmfT3KRUw"
        # SECRET_KEY="TZdqb575p-WNLXu_b-DMNrjKoMVkcsMeL9DS7WQKgmw"
        # CLIENT="2c168ebb-3514-4c3b-bb8d-19fcbe0c1ffa"
        # EAI_LOGIN_KEY= ACCESS_KEY:SECRET_KEY
        client = Client("https://snow-llmd-snow_dzmitry_llmd_bigcode_job.job.console.elementai.com", \
                     headers={"Authorization": "Bearer d4tVI9A5b0sYTEYLPL0yQcyMkkS81JJ2G4cmaHJkgro.RpYEN8Yeb0bcyfJ84bspg82JYaxhEKf6_zd8sACzZUM"})

    if opt.model_type == 'chatgpt':
        os.environ["OPENAI_API_KEY"] = open('/home/toolkit/repo_training_codellm/openai_api_key', 'r').read().strip()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.organization = "org-yPiFS0ZBT0mAkDrSnMPvfYap"
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.padding_side = 'left'
        client=None

    # NOTE: either specify eval_data to load with custom implementation or 
    #       dataset_path to load with hugging face functionality, but not both
    assert (opt.eval_data is not None) or opt.dataset_path is not None

    eval_dataset = src.data.ChatGPTDataset(
            data_path=opt.eval_data,
            tokenizer=tokenizer,
            passage_mode=opt.passage_mode,
            text_maxlen=opt.text_maxlength,
            num_of_examples=opt.num_of_eval_examples_per_gpu,
            model_type=opt.model_type,
        )
    logger.info(f'Loaded {len(eval_dataset)} validation examples from {opt.eval_data}')

    logger.info("Start eval")

    exactmatch, total = evaluate(eval_dataset, 
                                tokenizer, 
                                opt,
                                logger,
                                output_path,
                                start_idx,
                                client=client)

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