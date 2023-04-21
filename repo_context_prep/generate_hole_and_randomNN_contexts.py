import os
import json
import pickle
import argparse
import numpy as np
from utils import *
from context import *
from rule_config import *
from transformers import GPT2TokenizerFast, AutoTokenizer, AutoModel

def get_sorted_random_NN_contexts(retriever_context_obj):
    candidate_files = list(retriever_context_obj.parse_data.keys())
    chosen_files = np.random.choice(candidate_files, 100)
    #print(len(chosen_files))
    chosen_contexts = []
    for chosen_file in chosen_files:
      chosen_file_lines = open(chosen_file, encoding="utf8", errors='backslashreplace').readlines()
      if chosen_file == retrieval_context_obj.file:
        del chosen_file_lines[retrieval_context_obj.hole_pos[0]]
      chosen_line_idx = np.random.choice(len(chosen_file_lines))
      chosen_context = '\n'.join(chosen_file_lines[chosen_line_idx:])
      chosen_contexts.append(chosen_context)

    if chosen_contexts:
      sorted_contexts = retrieval_context_obj.get_similarity(chosen_contexts)
    else:
        sorted_contexts = []
    return sorted_contexts

def get_hole_context(file, hole_pos, num_of_prev_lines=2, num_of_post_lines=2):
    file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
    pre_end = hole_pos
    pre_start_line = hole_pos[0] - num_of_prev_lines
    if pre_start_line < 0:
        pre_start_line = 0
    pre_start = (pre_start_line, 0)
    pre_hole_context = get_string(file, pre_start, pre_end)

    post_hole_context = ""
    if num_of_post_lines > 0:    
        file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
        post_start_line = hole_pos[0] + 1
        if post_start_line < len(file_lines):
            post_end_line = hole_pos[0] + num_of_post_lines
            if post_end_line >= len(file_lines):
                post_end_line = len(file_lines) - 1
            post_start = (post_start_line, 0)
            post_end = (post_end_line, len(file_lines[post_end_line])) 
            post_hole_context = get_string(file, post_start, post_end)
    hole_context = post_hole_context + " " + pre_hole_context
    target_hole = file_lines[hole_pos[0]][hole_pos[1]:]
    return hole_context, target_hole

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--base_dir", type=str, default='/repo_data/repo_preprocessed_data/', help="base directory for the data")
  parser.add_argument("--data_split", type=str, default='train', help="data split")
  parser.add_argument("--repo_name", type=str, default='402d', help="name of the repo")
  parser.add_argument("--total_context_len", type=int, default=4072, help="total size of the rule context")

  return parser.parse_args()

if __name__ == '__main__':

    args = setup_args()

    #Fix seeds
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # get tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    #codebert tokenizer
    emb_model_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    emb_model_tokenizer.pad_token = emb_model_tokenizer.eos_token
    #codebert model
    emb_model = AutoModel.from_pretrained("microsoft/codebert-base")

    #directory for storing results
    input_data_dir = os.path.join(args.base_dir, args.data_split, args.repo_name)
    mod_data_split = 'medium_'  + args.data_split

    #get stored parsed data
    parsed_data_filename = os.path.join(input_data_dir, 'parsed_data')
    parse_data = pickle.load(open(parsed_data_filename, 'rb'))
    #get the holes
    hole_filename = os.path.join(input_data_dir, 'capped_hole_data_mod')
    hole_data = pickle.load(open(hole_filename, 'rb'))

    # file for storing hole and rule contexts
    f_out = open(os.path.join(args.base_dir, mod_data_split, args.repo_name, "hole_and_randomNN_contexts.json"), "w")

    # get all relevant files (in raw form)
    files = [os.path.join(dp, f) \
                for dp, dn, filenames in os.walk(input_data_dir) \
                for f in filenames \
                if os.path.splitext(f)[1] == '.java']

    print("Processing Repo: ", args.repo_name)
    for file in files:
        print("Processing File: ", file)  
        if file in hole_data:
            file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
            retrieval_context_obj = getRetrievalContext(tokenizer=emb_model_tokenizer,
                        file=file,
                        parse_data=parse_data,
                        emb_model=emb_model,
                        device='cuda:0')
            # iterate over all the holes in the file
            for (l,c) in hole_data[file]: # l = line no, c = character offset within line l
                hole_pos = (l, c)
                hole_identity = file + '_' + str(l) + '_' + str(c)
                hole_context, target_hole = get_hole_context(file, hole_pos)
                rule_contexts = []
                retrieval_context_obj.set_hole_pos(hole_pos)
                sorted_contexts = get_sorted_random_NN_contexts(retrieval_context_obj)
                if len(sorted_contexts) > 0:
                    for i in range(len(sorted_contexts)):
                        chosen_context = sorted_contexts[i]
                        context, context_len = get_codex_tokenized_string(tokenizer, chosen_context, args.total_context_len, type='front')
                        rule_score = 1 / (i + 1)
                        rule_contexts.append({'title': 'randomNN', 'text': context, 'score':rule_score})

                else:
                    rule_contexts = [{'title': 'randomNN', 'text': '', 'score': 1/(i+1)}]
                entry = {
                'id': hole_identity,
                'question': hole_context,
                'target': target_hole,
                'answers': [target_hole],
                'ctxs':rule_contexts,
                }

                f_out.write(json.dumps(entry))
                f_out.write("\n")
                f_out.flush()

    f_out.close()