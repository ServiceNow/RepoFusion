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
        model_type='codet5',
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

    def get_contexts(self, contexts, question=None):
        # the passages are already stored in sorted rule order.
        if self.passage_mode == 'pretrained' or self.passage_mode == 'finetuned':
            prior_context = contexts[16]
            if not prior_context['text'] and self.model_type == 'codegen':
                return []
            prior_context['text'], _ = self.truncate_rule_context(prior_context['text'], self.text_maxlen, truncation_strategy='front')
            return [prior_context]

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

        # the default codex context is always the last passage. This is based on the understanding that the encoded representations
        # of the rules are concatenated in the order of the passages. Therefore, the decoder attends to this context last such that
        # the model can learn to complete the hole starting from the default codex context.
        elif self.passage_mode == 'truncation-codex-last':
            codex_context = contexts.pop(16) # 16 is the index of the default codex context.
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
            codex_context = contexts.pop(16) # 16 is the index of the default codex context.
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
                # take only contexts before the codex context such that the codex context can fit in later.
                contexts_before_codex = modified_contexts[:(self.n_context - len(codex_parts))]
            else:
                # take only n_context codex parts starting from the back.
                codex_parts = codex_parts[-self.n_context:]
                contexts_before_codex = []
            for part in codex_parts:
                if len(part) > 0:
                    contexts_before_codex.append({'title': codex_context['title'], \
                                                'text': self.tokenizer.decode(part, skip_special_tokens=True), \
                                                'score': codex_context['score']})
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
            contexts = self.get_contexts(example['ctxs'], question)
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
        # if len(p['input_ids'][0])==512:
        #     print(f"Decoded:{tokenizer.decode(p['input_ids'][0])}, Passages:{text_passages}")
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
        # if index.item()==301:
        # print(f"passage: {text_passages[0]}, target: {string_target}, hole_id: {hole_id}, index: {index}")
        passage_ids, passage_masks = encode_passages(batch_text_passages=text_passages,
                                                     tokenizer=self.tokenizer,
                                                     max_length=self.text_maxlength,
                                                     model_type=self.model_type)
        return (index, target_ids, target_mask, passage_ids, passage_masks, hole_id)

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
    
class ChatGPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 num_of_examples=-1,
                 passage_mode='pretrained',
                 model_type='chatgpt',
                 text_maxlen=4096,
                 tokenizer=None):

        examples = []
        # each example is a json object that consists of data for a single hole. We load the json object later for efficiency.
        for dp, dn, filenames in os.walk(data_path):
            for f in filenames:
                if f == 'hole_and_rule_contexts.json':
                    data_path = os.path.join(dp, f)
                    #print('Loading data from {}'.format(data_path))
                    lines = open(data_path, 'r').readlines()
                    for i, line in enumerate(lines):
                        examples.append(line.strip())

        # if num_of_examples is specified, we only load the first num_of_examples examples.
        if num_of_examples > 0:
            examples = examples[:num_of_examples]
        self.examples = examples
        self.passage_mode = passage_mode
        self.model_type = model_type
        self.text_maxlen = text_maxlen
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

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

    def __getitem__(self, index):
        example = self.examples[index]
        example = json.loads(example)
        target = example['target']
        target, _ = self.truncate_rule_context(target, self.text_maxlen, truncation_strategy='back')

        if self.passage_mode == 'pretrained':
            prior_context = example['ctxs'][16]['text']
            if not prior_context:
                prompt = ''
            else:
                prompt, len_prompt = self.truncate_rule_context(prior_context, self.text_maxlen, truncation_strategy='front')
            #print(len_prompt)

        if self.passage_mode == 'toprule+prior':
            prior_context = example['ctxs'][16]['text']
            top_rule_context = example['ctxs'][0]['text']
            if not (top_rule_context and prior_context):
                prompt = ''
            else:
                if self.model_type == 'chatgpt':
                    top_rule_len = int(self.text_maxlen/2) - 1 # whitespace
                if self.model_type == 'starcoder':
                    top_rule_len = int(self.text_maxlen/2) - 3 # special tokens for FIM

                top_rule_context_text, len_top_rule_context = self.truncate_rule_context(top_rule_context, top_rule_len, truncation_strategy='back')
                prior_context_len = self.text_maxlen - len_top_rule_context
                prior_context_text, final_prior_context_len = self.truncate_rule_context(prior_context, prior_context_len, truncation_strategy='front')
                #print(self.text_maxlen, top_rule_len, len_top_rule_context, prior_context_len, final_prior_context_len)
                if self.model_type == 'chatgpt':
                    prompt = top_rule_context_text + " " + prior_context_text
                if self.model_type == 'starcoder':
                    prompt = "<fim_prefix>" + prior_context_text + "<fim_suffix>" + top_rule_context_text + "<fim_middle>"
        # print("prompt: ", prompt)
        # print("target last: ", target)
        # print("example id: ", example['id'])
        # print("index: ", index)
        return index, target, example['id'], prompt


class HoleContextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 num_of_examples=-1,
                 tokenizer=None):

        examples = []
        # each example is a json object that consists of data for a single hole. We load the json object later for efficiency.
        for dp, dn, filenames in os.walk(data_path):
            for f in filenames:
                if f == 'hole_and_rule_contexts.json':
                    data_path = os.path.join(dp, f)
                    #print('Loading data from {}'.format(data_path))
                    lines = open(data_path, 'r').readlines()
                    for i, line in enumerate(lines):
                        examples.append(line.strip())

        # if num_of_examples is specified, we only load the first num_of_examples examples.
        if num_of_examples > 0:
            examples = examples[:num_of_examples]
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        example = json.loads(example)
        hole_context = example['question']
        hole_id = example['id']
        return hole_context, hole_id

class HoleContextCollator(object):
    def __init__(self, tokenizer, maxlength=512):
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        
    def __call__(self, batch):
        hole_context_texts = [x[0] for x in batch]
        hole_id = [x[1] for x in batch]
        hole_context = self.tokenizer(hole_context_texts, \
                                        padding='max_length',
                                        return_tensors="pt",
                                        max_length=self.maxlength,
                                        truncation=True)
        return hole_context["input_ids"], hole_context["attention_mask"], hole_id
