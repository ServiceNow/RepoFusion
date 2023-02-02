# Copyright (c) Facebook, Inc. and its affiliates, and
# Copyright (c) ServiceNow and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains dataloaders and collators for the datasets used in the paper. The dataloaders contain the code for different repo context ordering strategies.
"""

import os
import torch
import random
import json
import numpy as np
import pickle

import datasets
from datasets.distributed import split_dataset_by_node
from pathlib import Path
import transformers

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        features_format_file=None,
        data_file_pattern=None,
        split=None,
        hf_datasets_cache_dir=None,
        hf_datasets_load_num_proc=None,
        n_context=None,
        global_rank=-1,
        world_size=-1,
        question_prefix='hole_context:',
        title_prefix='rule_name:',
        passage_prefix='rule_context:',
        passage_mode='truncation-direct',
        text_maxlen=512,
        tokenizer=None, 
        is_append_question=True, 
        num_of_examples=-1,
        model_type='codet5',
        write_hole_pp_mappings=False
    ):
        assert data_path
        if features_format_file is not None:
            data_path = Path(data_path)
            features_format_file = data_path / features_format_file
            ftrs = datasets.Features.from_dict(json.loads(Path(
                features_format_file
            ).read_text()))
            ds = datasets.load_dataset(
                str(data_path),
                data_files={split: split+'/'+data_file_pattern},
                split=split,
                features=ftrs,
                num_proc=hf_datasets_load_num_proc,
                cache_dir=hf_datasets_cache_dir
            )
            # NOTE: Is this is the bahaviour we want or  better assert for consistency
            #       between distributed params
            if global_rank == -1 or world_size == -1:
                global_rank = 0
                world_size = 1
            if world_size > 1:
                ds = split_dataset_by_node(ds, global_rank, world_size)
            self.ds = ds
            
            # if num_of_examples is specified, we only load the first num_of_examples examples.
            if num_of_examples > 0:
                self.ds = self.ds.select(range(num_of_examples))
            print(
                'Loaded {} examples with global rank {} and world size {}'.format(
                    len(self.ds), global_rank, world_size
            ))
            self.examples = None
        else:
            examples = []
            # each example is a json object that consists of data for a single hole. We load the json object later for efficiency.
            for dp, dn, filenames in os.walk(data_path):
                for f in filenames:
                    if f == 'hole_and_rule_contexts.json':
                        data_path = os.path.join(dp, f)
                        #print('Loading data from {}'.format(data_path))
                        lines = open(data_path, 'r').readlines()
                        for i, line in enumerate(lines):
                            # distribute the data across the ranks (GPUs).
                            if global_rank > -1 and not i % world_size == global_rank:
                                continue
                            examples.append(line.strip())
                        #data_path.close()

            # if num_of_examples is specified, we only load the first num_of_examples examples.
            if num_of_examples > 0:
                examples = examples[:num_of_examples]
            print('Loaded {} examples with global rank {} and world size {}'.format(len(examples), global_rank, world_size))
            self.examples = examples
            self.ds = None
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        # determines the way the passages are created for each hole.
        self.passage_mode = passage_mode
        self.text_maxlen = text_maxlen
        self.tokenizer = tokenizer
        self.is_append_question = is_append_question
        self.write_hole_pp_mappings = write_hole_pp_mappings
        if self.write_hole_pp_mappings:
            hole_pp_filename = os.path.join('/repo_data/repo_preprocessed_data/hole_pp_mappings', \
                                            str(self.text_maxlen) + '_' + str(self.n_context) + '_' + self.passage_mode + '.json')
            print(hole_pp_filename)
            self.hole_pp_map = open(hole_pp_filename, 'a')

        if self.passage_mode == 'pretrained' or self.passage_mode == 'finetuned' or self.passage_mode == 'toprule+prior':
            self.n_context = 1
            self.is_append_question = False
            self.title_prefix = ''
            self.passage_prefix = ''
            self.question_prefix = ''
            self.model_type = model_type

    def __len__(self):
        if self.ds is None:
             return len(self.examples)
        return len(self.ds)

    def divide_chunks(self, l, n):   
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_rule_context_length(self, question, rule_title):
        if rule_title == 'codex':
            truncation_strategy = 'front'
        else:
            truncation_strategy = 'back'
        net_string = self.title_prefix + " " + rule_title + " " + self.passage_prefix + " "
        rule_title_len = len(self.tokenizer(net_string).input_ids)
        if self.is_append_question:
            question = question + " "
            question_tokens_len = len(self.tokenizer(question).input_ids)
            rule_context_len = self.text_maxlen - (question_tokens_len + rule_title_len)
        else:
            rule_context_len = self.text_maxlen - rule_title_len
        return rule_context_len, truncation_strategy

    def truncate_rule_context(self, text, max_length, truncation_strategy='back'):
        text_tokens = self.tokenizer(text).input_ids
        if len(text_tokens) > max_length:
            if truncation_strategy == 'front':
                # take the last max_length tokens.
                text_tokens = text_tokens[-max_length:]
            else:
                # truncate the first max_length tokens.
                text_tokens = text_tokens[:max_length]
        return self.tokenizer.decode(text_tokens, skip_special_tokens=True), len(text_tokens)

    def truncate_contexts(self, contexts, question):
        for context in contexts:
            rule_context_len, rule_truncation_strategy = self.get_rule_context_length(question, context['title'])
            context['text'], _ = self.truncate_rule_context(context['text'], rule_context_len, rule_truncation_strategy)
        return contexts

    def get_contexts(self, contexts, question=None, hole_id=''):
        # the passages are already stored in sorted rule order.
        if self.passage_mode == 'pretrained' or self.passage_mode == 'finetuned':
            prior_context = contexts[16]
            if not prior_context['text'] and self.model_type == 'codegen':
                return []
            prior_context['text'], _ = self.truncate_rule_context(prior_context['text'], self.text_maxlen, truncation_strategy='front')
            return [prior_context]

        if self.passage_mode == 'repeated_toprule':
            top_rule_context = contexts[0]
            if not top_rule_context['text']:
                return []
            repeated_contexts = [top_rule_context] * self.n_context
            return self.truncate_contexts(repeated_contexts, question)

        if self.passage_mode == 'repeated_randomrule':
            idx = np.random.randint(0, len(contexts))
            random_rule_context = contexts[idx]
            if not random_rule_context['text']:
                return []
            repeated_contexts = [random_rule_context] * self.n_context
            return self.truncate_contexts(repeated_contexts, question)

        if self.passage_mode == 'repeated_priorrule':
            prior_rule_context = contexts[16]
            if not prior_rule_context['text']:
                return []
            repeated_contexts = [prior_rule_context] * self.n_context
            return self.truncate_contexts(repeated_contexts, question)

        if self.passage_mode == 'toprule+prior':
            prior_context = contexts[16]
            top_rule_context = contexts[0]
            if not (top_rule_context['text'] and prior_context['text']):
                return []
            if self.model_type == 'codegen' or self.model_type == 'codet5':
                top_rule_len = int(self.text_maxlen/2) - 1 # whitespace
            if self.model_type == 'santacoder':
                top_rule_len = int(self.text_maxlen/2) - 3 # special tokens for FIM

            top_rule_context_text, len_top_rule_context = self.truncate_rule_context(top_rule_context['text'], top_rule_len, truncation_strategy='back')
            prior_context_len = self.text_maxlen - len_top_rule_context
            prior_context_text, _ = self.truncate_rule_context(prior_context['text'], prior_context_len, truncation_strategy='front')

            if self.model_type == 'codegen' or self.model_type == 'codet5':
                net_context_text = top_rule_context_text + " " + prior_context_text
            if self.model_type == 'santacoder':
                net_context_text = "<fim-prefix>" + prior_context_text + "<fim-suffix>" + top_rule_context_text + "<fim-middle>"

            return [{'title': prior_context['title'], 'text': net_context_text, 'score': prior_context['score']}]

        if self.passage_mode == 'truncation-direct':
            return self.truncate_contexts(contexts[:self.n_context], question)

        # the prior context is always the last passage. This is based on the understanding that the encoded representations
        # of the rules are concatenated in the order of the passages. Therefore, the decoder attends to this context last such that
        # the model can learn to complete the hole starting from the prior context.
        elif self.passage_mode == 'truncation-codex-last':
            codex_context = contexts.pop(16) # 16 is the index of the prior context.
            contexts_without_codex = contexts[:(self.n_context-1)]
            contexts_without_codex.append(codex_context)
            return self.truncate_contexts(contexts_without_codex, question)

        # randomly shuffle the contexts. rule order is distorted here.
        elif self.passage_mode == 'truncation-random':
            random.shuffle(contexts)
            return self.truncate_contexts(contexts[:self.n_context], question)

        # split the non-empty contexts into chunks till it fits in context length in the sorted order.
        elif self.passage_mode == 'no-truncation-direct':
            modified_contexts = []
            for context in contexts:
                if context['text']:
                    tokens = self.tokenizer(context['text']).input_ids
                    rule_context_len, _ = self.get_rule_context_length(question, context['title'])
                    if rule_context_len > 0:
                        parts = list(self.divide_chunks(tokens, rule_context_len))
                    else:
                        parts = []
                    for part in parts:
                        if len(part) > 0:
                            modified_contexts.append({'title': context['title'], \
                                                    'text': self.tokenizer.decode(part, skip_special_tokens=True), \
                                                    'score': context['score']})
            return modified_contexts[:self.n_context]

        elif self.passage_mode == 'no-truncation-codex-last':          
            modified_contexts = []
            codex_context = contexts.pop(16) # 16 is the index of the prior context.
            if codex_context['text']:
                codex_tokens = self.tokenizer(codex_context['text']).input_ids
                rule_context_len, _ = self.get_rule_context_length(question, codex_context['title'])
                if rule_context_len > 0:
                    codex_parts = list(self.divide_chunks(codex_tokens, rule_context_len))
                else:
                    codex_parts = []
            else:
                codex_parts = []

            for context in contexts:
                if context['text']:
                    # Think how to do formatting here.
                    tokens = self.tokenizer(context['text']).input_ids
                    rule_context_len, _ = self.get_rule_context_length(question, context['title'])
                    if rule_context_len > 0:
                        parts = list(self.divide_chunks(tokens, rule_context_len))
                    else:
                        parts = []              
                    for part in parts:
                        if len(part) > 0:
                            modified_contexts.append({'title': context['title'], \
                                                    'text': self.tokenizer.decode(part, skip_special_tokens=True), \
                                                    'score': context['score']})

            if self.n_context >= len(codex_parts):
                # take only contexts before the  prior context such that the prior context can fit in later.
                contexts_before_codex = modified_contexts[:(self.n_context - len(codex_parts))]
            else:
                # take only n_context prior context parts starting from the back.
                codex_parts = codex_parts[-self.n_context:]
                contexts_before_codex = []
            for part in codex_parts:
                if len(part) > 0:
                    contexts_before_codex.append({'title': codex_context['title'], \
                                                'text': self.tokenizer.decode(part, skip_special_tokens=True), \
                                                'score': codex_context['score']})

            if self.write_hole_pp_mappings:
                prompt_proposals = [c['title'] for c in contexts_before_codex]
                entry = {"id": hole_id, "prompt_proposals": prompt_proposals}
                self.hole_pp_map.write(json.dumps(entry) + '\n')
                self.hole_pp_map.flush()

            return contexts_before_codex

    def get_formatted_passages_and_scores(self, contexts):
        if (self.passage_mode == 'finetuned' and self.model_type == 'codet5') or \
            (self.passage_mode == 'pretrained' and self.model_type == 'codegen') or \
            (self.passage_mode == 'toprule+prior'):
            passages = [c['text'] for c in contexts]
        elif (self.passage_mode == 'pretrained' and self.model_type == 'codet5'):
            passages = [c['text'] + '<extra_id_0>' for c in contexts]
        else:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"   
            passages = [f.format(c['title'], c['text']) for c in contexts]
        scores = [float(c['score']) for c in contexts]
        scores = torch.tensor(scores)
        return passages, scores

    def __getitem__(self, index):
        if self.ds is None:
            example = self.examples[index]
            example = json.loads(example)
        else:
            example = self.ds[index]
        question = self.question_prefix + " " + example['question']
        target = example['target']

        if 'ctxs' in example and self.n_context is not None:
            contexts = self.get_contexts(example['ctxs'], question, example['id'])
            if len(contexts) > 0:
                passages, scores = self.get_formatted_passages_and_scores(contexts)
            else:
                passages, scores = None, None
                # TODO(egrave): do we want to keep this?
                # if len(contexts) == 0:
                #     contexts = [question]
        else:
            passages, scores = None, None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores,
            'hole_id': example['id'],
        }

    def get_example(self, index):
        if self.ds is None:
            example = self.examples[index]
            example = json.loads(example)
            return example
        return  self.ds[index]

       
def encode_passages(batch_text_passages, tokenizer, max_length, model_type):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer(
                text_passages,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
                truncation=True,
            )
        if model_type == 'codet5':
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        if model_type == 'codegen' or model_type == 'santacoder':
            passage_ids.append(p['input_ids'])
            passage_masks.append(p['attention_mask'])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, is_append_question=True, model_type='codet5', passage_mode='toprule+prior'):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.is_append_question = is_append_question
        self.model_type = model_type
        self.passage_mode = passage_mode

    def truncate_from_left(self, text):
        tokens = self.tokenizer(text, truncation=False).input_ids
        if len(tokens) > self.text_maxlength:
            tokens = tokens[-self.text_maxlength:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        string_target = target
        hole_id = [ex['hole_id'] for ex in batch]
        target = self.tokenizer(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            appended_passages = []
            if self.is_append_question:
                for t in example['passages']:
                    # append the hole_context after the rule_context because we want completion to be conditioned 
                    # on the hole_context.
                    appended_passages.append(t + " " + example['question'])
            else:
                appended_passages = example['passages']
            if (self.passage_mode == 'pretrained' and self.model_type == 'codet5') or self.model_type == 'santacoder':
                appended_and_truncated_passages = appended_passages
            else:
                appended_and_truncated_passages = [self.truncate_from_left(p) for p in appended_passages]
            return appended_and_truncated_passages
        
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(batch_text_passages=text_passages,
                                                     tokenizer=self.tokenizer,
                                                     max_length=self.text_maxlength,
                                                     model_type=self.model_type)
        return (index, target_ids, target_mask, passage_ids, passage_masks, hole_id)
