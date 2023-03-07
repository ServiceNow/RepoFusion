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

class getAllRulesContexts():
    def __init__(self, parse_data=None, tokenizer=None, total_context_len=4072):
        super(getAllRulesContexts, self).__init__()
        self.parse_data = parse_data
        self.total_context_len = total_context_len
        self.tokenizer = tokenizer
        unsorted_rules = list(combined_to_index.keys())
        self.rules = [unsorted_rules[i] for i in rule_order]
        #print(self.rules)
        #print("Total # of Rules: ", len(self.rules))

    def get_rule_prompt(self, rule_context_obj=None, context_location = 'in_file', context_division_ratio=0.5):

        rule_context_obj.set_context_len(self.total_context_len)
        allocated_rule_context_len = int(rule_context_obj.get_context_len()*context_division_ratio)
        rule_context_obj.set_context_len(allocated_rule_context_len)

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

    def get_default_prompt(self, hole_pos=(0,0), file=''):

        default_context_obj = getContext(context_location='in_file',
                                    tokenizer=self.tokenizer,
                                    file=file,
                                    context_len=self.total_context_len,
                                    context_scope='pre',\
                                    context_type='lines',\
                                    top_k=-1)

        default_context_obj.set_hole_pos(hole_pos)
        default_prompt, default_prompt_len = default_context_obj.get_line_context()
        return default_prompt, default_prompt_len

    def get_hole_context(self, file, hole_pos, num_of_prev_lines=2, num_of_post_lines=2):
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

    def get_rule_contexts(self, file, hole_pos):
        file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
        rule_contexts = []
        for i in range(len(self.rules)):
            rule = self.rules[i]
            rule_score = 1/ (i + 1)
            if rule == 'codex':
                rule_context, _ = self.get_default_prompt(
                                        hole_pos=hole_pos,
                                        file=file
                                        )
            else:
                cl, ct, cf = rule.split('#')
                rule_specific_hyperparams = rule_hyperparams[ct]
                rule_util = getContext(context_location=cl, 
                                    context_type=ct, 
                                    parse_data=self.parse_data,
                                    file=file,
                                    top_k=int(rule_specific_hyperparams['top_k'][0]),
                                    rule_context_formatting=rule_specific_hyperparams['rule_context_formatting'][0],
                                    file_lines=file_lines,
                                    tokenizer=self.tokenizer)
                is_out_file = rule_util.is_out_files()
                if is_out_file or (not is_out_file and cl == 'in_file'):
                    rule_util.set_hole_pos(hole_pos)
                    rule_context, _ = self.get_rule_prompt(rule_context_obj=rule_util, \
                                                            context_location=cl, \
                                                            context_division_ratio=float(cf))
                else:
                    rule_context = ''

            rule_contexts.append({'title': rule, 'text': rule_context, 'score':rule_score})
        return rule_contexts

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
    hole_filename = os.path.join(input_data_dir, 'hole_data_mod')
    hole_data = pickle.load(open(hole_filename, 'rb'))

    # file for storing #empty rule contexts per hole.
    f = open(os.path.join(args.base_dir, "empty_rule_contexts.txt"), "a+")

    # file for storing hole and rule contexts
    f_out = open(os.path.join(input_data_dir, "hole_and_rule_contexts.json"), "w")

    # get all relevant files (in raw form)
    files = [os.path.join(dp, f) \
                for dp, dn, filenames in os.walk(input_data_dir) \
                for f in filenames \
                if os.path.splitext(f)[1] == '.java']

    all_rule_context_obj = getAllRulesContexts(parse_data=parse_data,\
                                            tokenizer=tokenizer, \
                                            total_context_len=args.total_context_len)
    print("Processing Repo: ", args.repo_name)
    total_count = 0
    # get the prompts for all files
    for file in files:
        if file in hole_data:
            # print(file)
            # go through the holes in the file
            for (l,c) in hole_data[file]: # l = line no, c = character offset within line l
                if total_count %100==0:
                    print("Total Holes:", total_count)
                hole_pos = (l, c)
                hole_context, target_hole = all_rule_context_obj.get_hole_context(file, hole_pos)
                rule_contexts = all_rule_context_obj.get_rule_contexts(file, hole_pos)
                # print("Hole context: ", hole_context)
                # print("Last Rule Context", rule_contexts[-1])
                # print(len(rule_contexts))
                empty_rule_contexts = [x for x in rule_contexts if x['text'] == '' ]
                #print("Empty Rule Contexts:", len(empty_rule_contexts))
                total_count += 1
                  # append to hole stats file

                entry = {
                'id': total_count,
                'question': hole_context,
                'target': target_hole,
                'answers': [target_hole],
                'ctxs': rule_contexts,
                }

                f_out.write(json.dumps(entry))
                f_out.write("\n")
                f_out.flush()

                # data_split, repo_name, filename, #holes in the file, hole, #empty rule contexts
                f.write(args.data_split + ", " + args.repo_name + ", " + file + ", " \
                        + str(len(hole_data[file])) + ", " \
                        + str(hole_pos) + ", " + str(len(empty_rule_contexts)))
                f.write("\n")

    f.close()
    f_out.close()