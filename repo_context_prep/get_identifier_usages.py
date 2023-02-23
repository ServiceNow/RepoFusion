import os
import pickle
import argparse
import collections
from utils import *

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--base_dir", type=str, default='rule_classifier_data/test', \
                            help="base directory for the data")
  parser.add_argument("--proj_name", type=str, default='wicketbits', \
                            help="name of the input repo")

  return parser.parse_args()

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED']=str(args.seed)

  input_data_path = os.path.join(args.base_dir, args.proj_name)
  os.makedirs(input_data_path, exist_ok=True)

  #get stored parsed data
  parsed_data_filename = os.path.join(input_data_path, 'parsed_data')
  parse_data = pickle.load(open(parsed_data_filename, 'rb'))

  files = [os.path.join(dp, f) \
            for dp, dn, filenames in os.walk(input_data_path) \
            for f in filenames \
            if os.path.splitext(f)[1] == '.java']

  identifier_file = collections.defaultdict(list)
  for file in files:
    file_identifiers = [get_string(file, x[0], x[1]).strip() for x in parse_data[file]['identifiers']]
    file_identifiers = list(set(file_identifiers))
    for iden in file_identifiers:
      identifier_file[iden].append(file)

  identifier_usage = collections.defaultdict(list)
  for iden, files in identifier_file.items():
    for file in files:
      iden_usage = [x for x in parse_data[file]['identifiers'] if get_string(file, x[0], x[1]).strip() == iden]
      identifier_usage[iden].append((file, iden_usage))

  print(len(identifier_usage.keys()))
  with open(os.path.join(input_data_path, 'identifier_usage'), 'wb') as f:
    pickle.dump(identifier_usage, f)