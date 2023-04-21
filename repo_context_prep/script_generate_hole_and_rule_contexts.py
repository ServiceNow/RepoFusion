import os
base_data_dir = '/repo_data/repo_preprocessed_data'
orig_data_split = 'val'
mod_data_split = 'medium_val'

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

commands = []

for proj in os.listdir(os.path.join(base_data_dir, mod_data_split)):
    proj_name = proj.strip()
    command = "python generate_hole_and_randomNN_contexts.py --repo_name " + proj_name \
                + " --base_dir " + base_data_dir + " --data_split " + orig_data_split
    commands.append(command)

# commands_list = list(split(commands, 2))
# for i, command_list in enumerate(commands_list):
#     with open("commands_gen_hole_and_bm25_contexts_" + orig_data_split + "_" + str(i), 'w') as f:
#         f.writelines("%s\n" % command for command in command_list)
#     f.close()
print(len(commands))
with open("commands_gen_hole_and_randomNN_contexts_" + orig_data_split, 'w') as f:
    f.writelines("%s\n" % command for command in commands)