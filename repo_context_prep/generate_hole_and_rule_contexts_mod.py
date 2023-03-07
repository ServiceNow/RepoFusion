import os
import json
import pickle
import argparse
from utils import *
from context import *
from rule_config import *
from transformers import GPT2TokenizerFast

rule_order = [5, 7, 6, 1, 20, 22, 2, 0, 25, 3, 23, 24, 28, 26, 4, 21, 62, 31, 27, 29, 30, 8, 34, 32, \
              10, 33, 9, 35, 13, 11, 12, 36, 46, 44, 16, 14, 49, 45, 48, 40, 38, 19, 15, 18, 39, 43, 47,\
               17, 42, 41, 58, 56, 57, 61, 59, 60, 37, 52, 50, 53, 55, 54, 51]

unsorted_rules = list(combined_to_index.keys())
rules = [unsorted_rules[i] for i in rule_order]

def get_rule_prompt(rule_context_obj=None):
    context_location = getattr(rule_context_obj, 'context_location')
    if context_location == 'in_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_in_file_context()
    if context_location == 'parent_class_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_parent_class_file_context()
    if context_location == 'import_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_import_file_context()
    if context_location == 'sibling_file' or context_location == 'reverse_sibling_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_sibling_file_context()
    if context_location == 'similar_name_file' or context_location == 'reverse_similar_name_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_similar_name_file_context()
    if context_location == 'child_class_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_child_class_file_context()
    if context_location == 'import_of_similar_name_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_similar_name_file_context()
    if context_location == 'import_of_parent_class_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_parent_class_file_context()
    if context_location == 'import_of_child_class_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_child_class_file_context()
    if context_location == 'import_of_sibling_file':
        rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_sibling_file_context()
    # if context_location == 'random_file':
    #     rule_prompt, rule_prompt_len = rule_context_obj.get_random_file_context()
    # if context_location == 'identifier_usage_file_random':
    #     rule_prompt, rule_prompt_len = rule_context_obj.get_identifier_usage_context_random()
    # if context_location == 'identifier_usage_file_NN':
    #     rule_prompt, rule_prompt_len = rule_context_obj.get_identifier_usage_context_nearest_neighbour()
    # if context_location == 'random_file_NN':
    #     rule_prompt, rule_prompt_len = rule_context_obj.get_random_file_context_nearest_neighbour()
    # if context_location == 'bm25':
    #     rule_prompt, rule_prompt_len = rule_context_obj.get_bm25_context()
    return rule_prompt, rule_prompt_len

def get_default_prompt(hole_pos, default_context_obj):
    default_context_obj.set_hole_pos(hole_pos)
    default_prompt, _ = default_context_obj.get_line_context()
    return default_prompt

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
  parser.add_argument("--repo_name", type=str, default='wiverson', help="name of the repo")
  parser.add_argument("--total_context_len", type=int, default=4072, help="total size of the rule context")

  return parser.parse_args()

if __name__ == '__main__':

    args = setup_args()

    #Fix seeds
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # get tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    #directory for storing results
    input_data_dir = os.path.join(args.base_dir, args.data_split, args.repo_name)

    #get stored parsed data
    parsed_data_filename = os.path.join(input_data_dir, 'parsed_data')
    parse_data = pickle.load(open(parsed_data_filename, 'rb'))
    #get the holes
    hole_filename = os.path.join(input_data_dir, 'capped_hole_data_mod')
    hole_data = pickle.load(open(hole_filename, 'rb'))

    # file for storing #empty rule contexts per hole.
    f = open(os.path.join(args.base_dir, "empty_rule_contexts_mod.txt"), "a+")

    # file for storing hole and rule contexts
    f_out = open(os.path.join(input_data_dir, "hole_and_rule_contexts.json"), "w")

    # get all relevant files (in raw form)
    files = [os.path.join(dp, f) \
                for dp, dn, filenames in os.walk(input_data_dir) \
                for f in filenames \
                if os.path.splitext(f)[1] == '.java']

    print("Processing Repo: ", args.repo_name)
    for file in files:
        print("Processing File: ", file)
        rule_contexts = {}
        hole_contexts = {}
        # get rule contexts for all the holes in a given file
        for i in range(len(rules)):
            rule = rules[i]
            #print("Processing Rule: ", rule)
            rule_score = 1/ (i + 1)
            if file in hole_data:
                file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
                # define the rule context object. Depends on the file
                if rule == 'codex':
                    rule_context_obj = getContext(context_location='in_file',
                                tokenizer=tokenizer,
                                file=file,
                                context_len=args.total_context_len,
                                context_scope='pre',\
                                context_type='lines',\
                                top_k=-1)
                else:
                    cl, ct, cf = rule.split('#')
                    rule_specific_hyperparams = rule_hyperparams[ct]
                    rule_context_obj = getContext(context_location = cl, \
                                    tokenizer=tokenizer, \
                                    file=file,\
                                    parse_data=parse_data,\
                                    context_type=ct,\
                                    top_k=int(rule_specific_hyperparams['top_k'][0]),\
                                    top_k_type='first',\
                                    rule_context_formatting=rule_specific_hyperparams['rule_context_formatting'][0],\
                                    file_lines=file_lines, \
                                    context_len=args.total_context_len)

                    is_out_file = rule_context_obj.is_out_files()

                # if the rule entries in parse_data arw empty, then append empty rule context.
                if rule != 'codex' and not is_out_file and cl != 'in_file':
                    # rule context for all the holes in the file will be empty
                    for (l,c) in hole_data[file]:
                        hole_identity_rule = file + '_' + str(l) + '_' + str(c) + '_' + rule
                        rule_contexts[hole_identity_rule] = {'title': rule, 'text': '', 'score':rule_score}
                        if hole_identity not in hole_contexts:
                            hole_context, target_hole = get_hole_context(file, (l, c))
                            hole_contexts[hole_identity] = {'target_hole': target_hole, 'hole_context': hole_context}
                    continue
                
                # iterate over all the holes in the file
                for (l,c) in hole_data[file]: # l = line no, c = character offset within line l
                    hole_pos = (l, c)
                    hole_identity = file + '_' + str(l) + '_' + str(c)
                    if hole_identity not in hole_contexts:
                        hole_context, target_hole = get_hole_context(file, hole_pos)
                        hole_contexts[hole_identity] = {'target_hole': target_hole, 'hole_context': hole_context}

                    if rule == 'codex':
                        rule_context = get_default_prompt(hole_pos, rule_context_obj)

                    else:
                        rule_context_obj.set_hole_pos(hole_pos)
                        rule_context_obj.set_context_len(args.total_context_len)
                        allocated_rule_context_len = int(rule_context_obj.get_context_len()*float(cf))
                        rule_context_obj.set_context_len(allocated_rule_context_len)
                        rule_context, _ = get_rule_prompt(rule_context_obj=rule_context_obj)

                    hole_identity_rule = hole_identity + '_' + rule
                    rule_contexts[hole_identity_rule] = {'title': rule, 'text': rule_context, 'score':rule_score}

        # write the hole and rule contexts for a file into a json file.
        for hole_identity in hole_contexts:
            hole_wise_rule_contexts = []
            for rule in rules:
                hole_identity_rule = hole_identity + '_' + rule
                hole_wise_rule_contexts.append(rule_contexts[hole_identity_rule])

            assert len(hole_wise_rule_contexts) == len(rules)
            entry = {
            'id': hole_identity,
            'question': hole_contexts[hole_identity]['hole_context'],
            'target': hole_contexts[hole_identity]['target_hole'],
            'answers': [hole_contexts[hole_identity]['target_hole']],
            'ctxs': hole_wise_rule_contexts,
            }

            f_out.write(json.dumps(entry))
            f_out.write("\n")
            f_out.flush()

            empty_rule_contexts = [x for x in hole_wise_rule_contexts if x['text'] == '' ]
            # data_split, repo_name, filename, #holes in the file, hole, #empty rule contexts
            f.write(args.data_split + ", " + args.repo_name + ", " + hole_identity + ", " \
                    + str(len(empty_rule_contexts)))
            f.write("\n")

    f.close()
    f_out.close()