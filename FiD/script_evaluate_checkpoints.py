import os
base_data_dir = '/repo_data/repo_FID'
checkpoint_dir = os.path.join(base_data_dir, 'checkpoints')
checkpoints = os.listdir(checkpoint_dir)

other_checkpoints = [
    'codet5base_768_32_linear_no-truncation-direct_True',
    'codet5base_768_32_linear_no-truncation-codex-last_True'
]

commands = []
for checkpoint in checkpoints:
    if checkpoint.startswith('mod_') or checkpoint in other_checkpoints:
        command = "./run.sh -f FiD/test_reader.py --per_gpu_batch_size=1 --dataset_path=/repo_data/repo_preprocessed_data --num_of_eval_examples_per_gpu=-1 " \
                + "--eval_split_name=random_val "\
                + "--output_dir=" + os.path.join(base_data_dir, "checkpoint_evaluations") + " "\
                + "--passage_mode=no-truncation-codex-last " \
                + "--n_context=63 "\
                + "--model_max_length=512 "\
                + "--text_maxlength=512 "\
                + "--trained_model_path=" + os.path.join(checkpoint_dir, checkpoint) + " "\
                + "--trained_model_load_type=latest "\
                + "--name=" + checkpoint
        commands.append(command)


with open("commands_evaluate_checkpoints", 'w') as f:
  f.writelines("%s\n" % command for command in commands)
f.close()