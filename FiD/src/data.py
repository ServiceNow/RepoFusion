# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import random
import json
import numpy as np

import datasets
from datasets.distributed import split_dataset_by_node
from pathlib import Path

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
                self.ds = split_dataset_by_node(ds[split], global_rank, world_size)
            else:
                self.ds = ds[split]
            
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
        print('Loaded {} examples with global rank {} and world size {}'.format(len(examples), global_rank, world_size))
        self.examples = examples
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        # determines the way the passages are created for each hole.
        self.passage_mode = passage_mode
        self.text_maxlen = text_maxlen
        self.tokenizer = tokenizer
        self.is_append_question = is_append_question

    def __len__(self):
        if self.ds is None:
             return len(self.examples)
        return len(self.ds)

    def divide_chunks(self, l, n):   
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_contexts(self, contexts, question=None):
        # the passages are already stored in sorted rule order.

        if self.passage_mode == 'truncation-direct':
            return contexts[:self.n_context]

        # the default codex context is always the last passage. This is based on the understanding that the encoded representations
        # of the rules are concatenated in the order of the passages. Therefore, the decoder attends to this context last such that
        # the model can learn to complete the hole starting from the default codex context.
        elif self.passage_mode == 'truncation-codex-last':
            codex_context = contexts.pop(16) # 16 is the index of the default codex context.
            contexts.append(codex_context)
            return contexts[:self.n_context]

        # randomly shuffle the contexts. rule order is distorted here.
        elif self.passage_mode == 'truncation-random':
            random.shuffle(contexts)
            return contexts[:self.n_context]

        # split the non-empty contexts into chunks till it fits in context length in the sorted order.
        elif self.passage_mode == 'no-truncation-direct':
            if self.is_append_question:
                question_tokens_len = len(self.tokenizer(question)['input_ids'])
                rule_context_len = self.text_maxlen - question_tokens_len
            else:
                rule_context_len = self.text_maxlen
            modified_contexts = []
            for context in contexts:
                if context['text']:
                    tokens = self.tokenizer(context['text'])['input_ids']
                    parts = list(self.divide_chunks(tokens, rule_context_len))
                    for part in parts:
                        modified_contexts.append({'title': context['title'], 'text': self.tokenizer.decode(part), 'score': context['score']})
            return modified_contexts[:self.n_context]

        elif self.passage_mode == 'no-truncation-codex-last':
            if self.is_append_question:
                question_tokens_len = len(self.tokenizer(question)['input_ids'])
                rule_context_len = self.text_maxlen - question_tokens_len
            else:
                rule_context_len = self.text_maxlen           
            modified_contexts = []
            count = 0
            for context in contexts:
                count += 1
                if context['text']:
                    tokens = self.tokenizer(context['text'])['input_ids']
                    parts = list(self.divide_chunks(tokens, rule_context_len))
                    if count == 17: # 16 + 1 as count is incremented before the if condition.
                        codex_parts = parts
                        continue                
                    for part in parts:
                        modified_contexts.append({'title': context['title'], 'text': self.tokenizer.decode(part), 'score': context['score']})
            
            contexts_before_codex = modified_contexts[:(self.n_context-len(codex_parts))]
            for part in codex_parts:
                contexts_before_codex.append({'title': context['title'], 'text': self.tokenizer.decode(part), 'score': context['score']})
            return contexts_before_codex

    def __getitem__(self, index):
        if self.ds is None:
            example = self.examples[index]
            example = json.loads(example)
        else:
            example = self.ds[index]
        question = self.question_prefix + " " + example['question']
        target = example['target']

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = self.get_contexts(example['ctxs'], question)
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    # def sort_data(self):
    #     if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
    #         return
    #     for ex in self.data:
    #         ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        if self.ds is None:
            example = self.examples[index]
            example = json.loads(example)
            return example
        return  self.ds[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True,
            # truncate from the beginning to keep the end of the passage which is the hole context.
            truncation_side='left',
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, is_append_question=True):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.is_append_question = is_append_question

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
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
                    # append the hole_context after the rule_context because we want completion to be conditioned on the hole_context.
                    appended_passages.append(t + " " + example['question'])
            else:
                appended_passages = example['passages']
            return appended_passages
        
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

# def load_data(data_path=None, global_rank=-1, world_size=-1):
#     assert data_path
#     examples = []
#     for dp, dn, filenames in os.walk(data_path):
#       for f in filenames:
#         if f == 'hole_and_rule_contexts.json':
#             data_path = os.path.join(dp, f)
#             print('Loading data from {}'.format(data_path))
#             data = open(data_path, 'r')
#             for k, example in enumerate(data):
#                 if global_rank > -1 and not k%world_size==global_rank:
#                     continue
#                 if data_path is not None:
#                     example = json.loads(example)
#                 if not 'id' in example:
#                     example['id'] = k
#                 for c in example['ctxs']:
#                     if not 'score' in c:
#                         c['score'] = 1.0 / (k + 1)
#                 examples.append(example)
#             ## egrave: is this needed?
#             if data_path is not None and data_path.endswith('.json'):
#                 data.close()
#         print('Loaded {} examples'.format(len(examples)))
#     return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer(
            question,
            padding='max_length',
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer(
            [x[1] for x in batch],
            padding='max_length',
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
