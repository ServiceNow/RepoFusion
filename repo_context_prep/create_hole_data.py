import numpy as np
import os
import argparse
import pickle
'''
For each line in the repo (which is not blank or comment) choose the midpoint of those lines (character wise not token wise) as hole position.
'''

def choose_holes(project_lines, comments):
  data = {}
  count = 0
  repeated_holes = 0
  chosen_lines = []

  selected_lines = np.arange(0, len(project_lines))

  for proj_line_id in selected_lines:
    file, file_line_id, line = project_lines[proj_line_id]
    # removing leading and trailing whitespaces
    line = line.strip()
    # omitting comments and empty lines
    if line and not (np.any([line.startswith(comment) for comment in comments])):
      if proj_line_id in chosen_lines:
        repeated_holes+=1
      else:
        chosen_lines.append(proj_line_id)
      count+=1
      #get holes from the middle of the lines
      chosen_position = np.random.randint(0, len(line))
      # mid_point = int(len(line)/2)
      # chosen_position = mid_point
      chosen_position = np.random.randint(0, len(line))

      if file in data:
        data[file].append((file_line_id, chosen_position))
      else:
        data[file] = [(file_line_id, chosen_position)]

  return data, len(chosen_lines), len(data)

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--base_dir", type=str, default='/home/toolkit/repo_preprocessed_data', \
                            help="base directory for the data")
  parser.add_argument("--data_split", type=str, default='train', \
                            help="data split to store the data")
  parser.add_argument("--language", type=str, default='java', help="java, cpp")
  parser.add_argument("--proj_name", type=str, default='rbudde', \
                            help="name of the input repo")

  return parser.parse_args()

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED']=str(args.seed)

  if args.language == 'java':
    file_extensions = ['.java']
    comments = ['*', '/']
  if args.language == 'lua':
    file_extensions = ['.lua']
    comments = ['--']
  if args.language == 'cpp':
    file_extensions == ['.cc', '.cpp', '.h']
    comments = ['/']

  data_path = os.path.join(args.base_dir, args.data_split, args.proj_name)

  files = []
  for dp, dn, filenames in os.walk(data_path):
    for f in filenames:
      if f == 'unique_files.txt':
        unique_files = open(os.path.join(dp, f), 'r').readlines()
        unique_files = [f.strip() for f in unique_files]
      for file_ext in file_extensions:
        if os.path.splitext(f)[1] == file_ext:
          files.append(os.path.join(dp, f))

  project_lines = []
  for file in files:
    if file in unique_files:
      file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
      parsed_file_lines = []
      for l in range(len(file_lines)):
        line = file_lines[l]
        parsed_file_lines.append((file, l, line))
      project_lines.extend(parsed_file_lines)
  num_of_lines_in_proj = len(project_lines) # number of lines in the project
  data, num_of_holes, num_of_files = choose_holes(project_lines, comments=comments)

  # write hole data
  with open(os.path.join(data_path, 'hole_data'), 'wb') as f:
    pickle.dump(data, f)

  # append to hole stats file
  f = open(os.path.join(args.base_dir, "repo_stats.txt"), "a+")
  # data_split, repo_name, #unique_files, #all files, #lines, #holes
  f.write(args.data_split + ", " + args.proj_name + ", " + str(num_of_files) + ", " \
                      + str(len(files)) + ", " \
                      + str(num_of_lines_in_proj) + ", " + str(num_of_holes))
  f.write("\n")
  f.close()
