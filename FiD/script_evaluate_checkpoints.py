import os

base_data_dir = '/repo_data/repo_FID'
# checkpoint_dir = os.path.join(base_data_dir, 'checkpoints')
# checkpoints = os.listdir(checkpoint_dir)

other_checkpoints = [
    # 'codet5base_768_32_linear_no-truncation-direct_True',
    # 'codet5base_768_32_linear_no-truncation-codex-last_True'
]
directory = os.path.join(base_data_dir, "checkpoints", "mod_finetuned_codet5base_768_32_linear_2.5e-5_no-truncation-direct_True")
checkpoints = os.listdir(os.path.join(directory, "checkpoint"))
latest = 72500
before = 199500

for i in range(latest, latest-10000, -500):
    other_checkpoints.append(str(i))
other_checkpoints.append(str(before))
commands = []
for checkpoint in other_checkpoints:
    #if checkpoint.startswith('mod_') or checkpoint in other_checkpoints:
        command = "./run.sh -f FiD/test_reader.py --per_gpu_batch_size=1 --dataset_path=/repo_data/repo_preprocessed_data --num_of_eval_examples_per_gpu=-1 " \
                + "--eval_split_name=random_val "\
                + "--output_dir=" + os.path.join(base_data_dir, "checkpoint_evaluations") + " "\
                + "--passage_mode=no-truncation-codex-last " \
                + "--n_context=63 "\
                + "--model_max_length=512 "\
                + "--text_maxlength=512 "\
                + "--trained_model_path=" + directory + " "\
                + "--trained_model_load_type=step-" + checkpoint + " "\
                + "--name=2_" + checkpoint
        commands.append(command)


with open("commands_evaluate_checkpoints_2", 'w') as f:
  f.writelines("%s\n" % command for command in commands)
f.close()