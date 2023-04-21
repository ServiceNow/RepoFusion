import os
from utils import *
import numpy as np
import torch
from transformers import AutoModel
from rank_bm25 import BM25Okapi

class getRetrievalContext():

  def __init__(self, file='', parse_data=None, identifier_usage_data=None, tokenizer=None, emb_model=None, device=''):

    super(getRetrievalContext, self).__init__()
    self.tokenizer = tokenizer
    self.parse_data = parse_data
    self.file = file
    self.identifier_usage_data = identifier_usage_data
    self.emb_model = emb_model
    self.device = device
    if self.emb_model is not None:
      self.emb_model = self.emb_model.to(self.device)

  def set_hole_pos(self, hole_pos):
    self.hole_pos = hole_pos

  def get_window_start_and_end(self, line, file, num_of_prev_lines, num_of_post_lines):
    start_line = line - num_of_prev_lines
    if start_line < 0:
      start_line = 0
    end_line = line + num_of_post_lines
    file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
    if end_line >= len(file_lines):
      end_line = len(file_lines) - 1
    return start_line, end_line, file_lines

  def get_sorted_identifiers_near_the_hole(self, num_of_prev_lines=2, num_of_post_lines=2):
    hole_line = self.hole_pos[0]
    hole_window_start_line, hole_window_end_line, _ = self.get_window_start_and_end(hole_line, self.file, num_of_prev_lines, num_of_post_lines)
    candidate_identifiers = []
    all_file_identifiers = self.parse_data[self.file]['identifiers']
    for iden in all_file_identifiers:
      iden_line = iden[0][0]
      # take all identifiers appearing in the hole window region
      if (hole_window_start_line <= iden_line <= hole_window_end_line):
        if (iden_line != hole_line) or (iden_line == hole_line and iden[1][1] < self.hole_pos[1]):
          candidate_identifiers.append((iden, np.abs(iden_line-hole_line)))
    sorted_candidate_identifiers = sorted(candidate_identifiers, key=lambda x: x[1])
    sorted_candidate_identifiers = [get_string(self.file, x[0], x[1]).strip() for (x, y) in sorted_candidate_identifiers]
    sorted_candidate_identifiers = list(set(sorted_candidate_identifiers))
    return sorted_candidate_identifiers

  def get_window(self, att, file, is_hole=False, num_of_prev_lines=2, num_of_post_lines=2):
    if is_hole:
      return self.get_hole_window(num_of_prev_lines, num_of_post_lines)

    att_line = att[0][0]
    start_line, end_line, file_lines = self.get_window_start_and_end(att_line, file, num_of_prev_lines, num_of_post_lines)

    # if attribute comes from the hole file and the att_window overlaps with the hole, skip it.
    if (file == self.file) and (start_line <= self.hole_pos[0] <= end_line):
      return ''
    else:
      start = (start_line, 0)
      end = (end_line, len(file_lines[end_line]))
      att_window = get_string(file, start, end)
      return att_window

  def get_hole_window(self, num_of_prev_lines=2, num_of_post_lines=2):
    '''
    return the window of context around a usage
    '''
    file = self.file
    pos = self.hole_pos

    pre_end = pos
    pre_start_line = pos[0] - num_of_prev_lines
    if pre_start_line < 0:
      pre_start_line = 0
    pre_start = (pre_start_line, 0)
    pre_hole_context = get_string(file, pre_start, pre_end)

    post_hole_context = ""
    if num_of_post_lines > 0:
      file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
      post_start_line = pos[0] + 1
      if post_start_line < len(file_lines):
        post_end_line = pos[0] + num_of_post_lines
        if post_end_line >= len(file_lines):
          post_end_line = len(file_lines) - 1
        post_start = (post_start_line, 0)
        post_end = (post_end_line, len(file_lines[post_end_line]))
        post_hole_context = get_string(file, post_start, post_end)
    hole_window = post_hole_context + " " + pre_hole_context
    return hole_window

  def find_usage_windows(self, query_iden, file, usages):
    usage_windows = []
    for usage in usages:
      usage_window = self.get_window(usage, file, is_hole=False)
      if usage_window:
          usage_windows.append(usage_window)
    return usage_windows

  def get_identifier_usage_windows(self):
    # get all identifiers in the hole window region excluding the hole itself.
    candidate_identifiers = self.get_sorted_identifiers_near_the_hole()
    # if no identifiers found in the hole window
    if not candidate_identifiers:
      return []
    iden_usage_windows = []
    # get the usage context of the identifier from all files, go to next identifier only if nothing is returned from the first
    for iden in candidate_identifiers:
      for file, usages in self.identifier_usage_data[iden]:
        iden_usage_windows.extend(self.find_usage_windows(iden, file, usages))
      if iden_usage_windows:
        break
    return iden_usage_windows

  def get_representation(self, context):
    outputs = self.emb_model(**context)
    try:
        representation = outputs.pooler_output
    except:
        representation = outputs.last_hidden_state[:, 0]
    return representation

  def get_context_embedding(self, context_str):
    context = self.tokenizer(context_str, truncation=True, padding='max_length', return_tensors="pt").to(self.device)
    context_embedding = self.get_representation(context)
    return context_embedding

  def get_similarity(self, iden_usage_windows):
    if len(iden_usage_windows) > 64:
        np.random.shuffle(iden_usage_windows)
        iden_usage_windows = iden_usage_windows[:64]
    sorted_iden_usage_windows = []
    hole_window = self.get_hole_window()
    hole_repr = self.get_context_embedding(hole_window)
    iden_repr = self.get_context_embedding(iden_usage_windows)
    #print(hole_repr.shape, iden_repr.shape)
    iden_scores = torch.squeeze(torch.matmul(hole_repr, iden_repr.transpose(0, 1)), dim=0)
    # print(hole_repr.shape, iden_repr.shape, iden_scores.shape)
    # print(iden_scores)
    for i in range(len(iden_scores)):
      iden_window = iden_usage_windows[i]
      iden_score = iden_scores[i]
      #print(iden_score)
      sorted_iden_usage_windows.append((iden_window, iden_score))

    sorted_iden_usage_windows = sorted(sorted_iden_usage_windows, key=lambda x: x[1], reverse=True)
    sorted_iden_usage_windows = [x for (x, y) in sorted_iden_usage_windows]
    return sorted_iden_usage_windows

  def get_bm25_sorted_contexts(self):
    all_files = list(self.parse_data.keys())
    #chosen_file = np.random.choice(candidate_files)
    tokenized_corpus = []
    corpus = []
    for file in all_files:
      if file == self.file:
        file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
        del file_lines[self.hole_pos[0]]
        file_content = '\n'.join(file_lines)
      else:
        file_content = open(file, encoding="utf8", errors='backslashreplace').read()
      corpus.append(file_content)
      tokenized_corpus.append(file_content.split(" "))
    bm25 = BM25Okapi(tokenized_corpus)
    query = self.get_hole_window()
    # print(self.file, self.hole_pos, query, len(corpus))
    tokenized_query = query.split(" ")
    sorted_contexts = bm25.get_top_n(tokenized_query, corpus, n=100)
    return sorted_contexts






        # if iden_usage_windows:
        #   for iden_usage_window in iden_usage_windows:
        #     iden_usage_score = self.get_similarity(iden_usage_window, hole_context)
        #     self.update_top_scores(iden_usage_score, usage_scores, top_k=10)


  # def get_relevant_usage_contexts(self, usages):
  #   hole_window = get_context(hole)
  #   hole_repr = get_context_embedding(hole_window)
  #   usage_scores = {}
  #   for usage in usages:
  #     usage_window = get_context(usage)
  #     usage_repr = get_context_embedding(usage_window)
  #     usage_score = dot_product(hole_repr, usage_repr)
  #     usage_scores[usage_window] = usage_score
  #   sorted_usage_contexts = sorted(usage_scores, lambda x: x[1], reverse=True)
  #   return sorted_usage_contexts

  # def get_usages_from_context_location(self, hole_attributes, context_location):
  #   cl_files = self.get_relevant_files(context_location)
  #   for hole_att in hole_attributes:
  #     for cl_file in cl_files:
  #       print(cl_file)
  #       file_attributes = self.parse_data[cl_file]['identifiers']
  #       att_usages = find_usages(hole_att, self.file, file_attributes, cl_file)
  #       if not att_usages:
  #         continue
  #       else:
  #         return (cl_file, att_usages)

  # def get_all_usages(self, hole_attributes):
  #   usages = {}
  #   for context_location in context_location_conversion.keys():
  #     print(context_location)
  #     usages[context_location] = self.get_usages_from_context_location(hole_attributes, context_location)
  #   return usages
