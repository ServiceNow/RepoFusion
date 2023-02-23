import os
base_data_dir = '/home/toolkit/repo_preprocessed_data'
data_splits = os.listdir(base_data_dir)

commands_parse = []
commands_hole = []
for data_split in data_splits:
  for proj in os.listdir(os.path.join(base_data_dir, data_split)):
    proj_name = proj.strip()
    command = "python create_hole_data.py --proj_name " + proj_name \
              + " --base_dir " + base_data_dir + " --data_split " + data_split
    commands_hole.append(command)
    command = "python parse_tree.py --proj_name " + proj_name \
              + " --base_dir " + os.path.join(base_data_dir, data_split)
    commands_parse.append(command)

with open("commands_gen_and_preprocess_hole", 'w') as f:
  f.writelines("%s\n" % command for command in commands_hole)
f.close()

with open("commands_gen_and_preprocess_parse", 'w') as f:
  f.writelines("%s\n" % command for command in commands_parse)
f.close()