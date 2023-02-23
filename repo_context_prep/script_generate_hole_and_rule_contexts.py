import os
base_data_dir = '/repo_data/repo_preprocessed_data'
data_split = 'train'

commands = []

for proj in os.listdir(os.path.join(base_data_dir, data_split)):
    proj_name = proj.strip()
    command = "python generate_hole_and_repo_contexts.py --repo_name " + proj_name \
                + " --base_dir " + base_data_dir + " --data_split " + data_split
    commands.append(command)

with open("commands_gen_hole_and_repo_contexts", 'w') as f:
  f.writelines("%s\n" % command for command in commands)
f.close()