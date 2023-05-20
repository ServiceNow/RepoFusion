## Intro
This folder contais code to prepare dataset and fine tune `codet5` models on the next token prediction objective on the Java subset of `The Stack`. The dataset post processed for `Santa Coder` training is used as source (https://huggingface.co/datasets/bigcode/stack-dedup-alt-comments) as well as file names from repositories  used to create `Stack Repo` dataset to exclude them from data.

## Dataset preparation
Create a config file from `../conf/codet5_finetune/base.yaml` and store it in the same folder. Only parameters under `Data preparation params` has effect on data preparation. Set the following parmeters to local dataset path:
- `filename_1K_20plus_file_list` - the path to a file `[Stack Repo dataset path]/1K_20plus_file_list.parquet` with the list of files used in `Stack Repo`
- `filename_the_stack11_dedup_alt_comments` - local path to  `stack-dedup-alt-comments`
- `path_java_filtered_subset_root` - destination path for the fine tuning dataset
- the rest of the parameters control split sizes, seed etc.


Set `PYTHONPATH` to the this repository root

From this repository root run:
```
python codet5_finetune/prep_data.py conf/codet5_finetune/[your config file].yaml
```

