# Copyright (c) Facebook, Inc. and its affiliates, and
# Copyright (c) ServiceNow and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configargparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000) 
        self.parser.add_argument('--total_steps', type=int, default=500000) 
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_false', help='save results, default is True')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', 
                        help='save dataset with cross-attention scores, default is False')
        self.parser.add_argument('--trained_model_path', type=str, \
                            default=None, help='path of the model to be evaluated')
        self.parser.add_argument('--trained_model_load_type', type=str, default='best_dev', help='best_dev or latest')
        self.parser.add_argument('--output_dir', type=str, \
                            default='../outputs', \
                                help='path of the output directory')
        self.parser.add_argument('--eval_print_freq', type=int, default=100, help='print frequency')


    def add_reader_options(self):
        # NOTE: either specify train_data and eval_data to load with custom implementation or 
        # dataset_path to load with hugging face functionality, but not both
        self.parser.add_argument('--train_data', type=str, default='../the-stack-repo/train/', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='../the-stack-repo/val/', help='path of eval data')
        self.parser.add_argument(
            '--dataset_path', type=str, default=None, 
            help='path to a dataset to be loaded with hugging face funcionality'
        )
        self.parser.add_argument(
            '--train_split_name', type=str, default='train',
            help='training split name to be loaded with hugging face funcionality'
        )
        self.parser.add_argument(
            '--eval_split_name', type=str, default='val',
            help='eval split name to be loaded with hugging face funcionality'
        )
        self.parser.add_argument(
            '--features_format_file',
            type=str, default='hf_dataset_features_format.json',
            help='dataset format file path either relative to dataset_path or absolute'
        )
        self.parser.add_argument(
            '--data_file_pattern',
            type=str, default='*/hole_and_PP_contexts.json',
            help='data files pattern to look for inside split folder'
        )
        self.parser.add_argument(
            '--hf_datasets_load_num_proc', type=int, default=1, 
            help='number of processes to use to load dataset, speeds up creation of the dataset cahche'
        )
        self.parser.add_argument(
            '--hf_datasets_cache_dir',
            type=str, default='/repo_data/hf_datasets_cashe',
            help='forder path to use as datasets chache, None will result in default'
        )
        self.parser.add_argument('--model_type', type=str, default='codet5', help='type of the model')
        self.parser.add_argument('--model_name', type=str, default='Salesforce/codet5', help='model name without size')
        self.parser.add_argument('--model_max_length', type=int, default=512, help='model_max_length for tokenizer')
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--text_maxlength', type=int, default=512,  
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=512, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=63)
        self.parser.add_argument('--is_append_question', action='store_false', help='whether to append question to passage')
        self.parser.add_argument('--write_hole_pp_mappings', action='store_true', help='whether to write hole PP mappings or not')
        self.parser.add_argument('--passage_mode', type=str, default = 'truncation-direct', \
                                    help = 'different modes of treating the passages. Options are truncation-direct, no-truncation-direct, \
                                    truncation-random, no-truncation-codex-last, truncation-codex-last')
        self.parser.add_argument('--num_of_eval_examples_per_gpu', type=int, default=5000,  
                        help='maximum number of examples to take from the eval dataset')

    def add_retriever_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true', 
                        help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40, 
                        help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200, 
                        help='maximum number of tokens in passages')
        self.parser.add_argument('--no_question_mask', action='store_true')
        self.parser.add_argument('--no_passage_mask', action='store_true')
        self.parser.add_argument('--extract_cls', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)


    def initialize_parser(self):
        self.parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')
        # basic parameters
        self.parser.add_argument('--name', type=str, default='debug', help='name of the experiment')
        self.parser.add_argument('--initialize_from_pretrained', action='store_false', #default value is True
                        help='whether to initialize from a pretrained model or not. When this is False,\
                        the model is initialized from the finetuned_model_path')
        self.parser.add_argument('--checkpoint_dir', type=str, default='/repo_data/repo_FID/checkpoints/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--finetuned_model_path', type=str, \
                                    default='../trained_checkpoints/finetuned_codet5base_512', \
                                        help='path of the finetuned model')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--eval_loss_freq', type=int, default=5000,
                        help='evaluate model loss every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=500,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_em_freq', type=int, default=5000,
                        help='evaluate model EM every <eval_em_freq> steps')


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self, args=None):
        opt = self.parser.parse_args(args)
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
