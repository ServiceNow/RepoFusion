## Intro
This folder contais code to prepare dataset and fine tune `codet5` models on the next token prediction (NTP) objective on the Java subset of `The Stack`. The dataset post processed for `Santa Coder` training is used as source (https://huggingface.co/datasets/bigcode/stack-dedup-alt-comments) as well as file names from repositories  used to create `Stack Repo` dataset to exclude them from data.

## Dataset preparation
Modify the cofing file `../conf/codet5_finetune/base.yaml`. Only parameters under `Data preparation params` has effect on data preparation. Set the following parmeters to local dataset path:
- `filename_1K_20plus_file_list` - the path to a file `[this repo root]/codet5_finetune/1K_20plus_file_list.parquet` with the list of files to exlude, those files are partially used in `Stack Repo`
- `filename_the_stack11_dedup_alt_comments` - local path to  `stack-dedup-alt-comments`
- `path_java_filtered_subset_root` - destination path for the fine tuning dataset
- the rest of the parameters control split sizes, seed etc.


Set `PYTHONPATH` to the this repository root

From this repository root run:
```
python codet5_finetune/prep_data.py
```

## Fineune codet5 model on prepared data
Assuming the previous step is done, modify the following parameter in the cofing file `../conf/codet5_finetune/base.yaml`:
- `model_dir_base` - the distination path base folder for checkpoints and tensorboard to be stored

Assuming on a machine with 2 A100-80Gb for finetuning base model and with 4 for large.

Set `PYTHONPATH` to the this repository root

From this repository root run the following command to finetune the base model:
```
accelerate launch --multi_gpu --num_processes=2 codet5_finetune/train.py conf/codet5_finetune/train_tAll_v10k_sl512_nep1_bspd32_dn2_graccs2_lr4e-5_wup100_wd005.yaml
```

From this repository root run the following command to finetune the base model:
```
accelerate launch --multi_gpu --num_processes=4 codet5_finetune/train.py conf/codet5_finetune/train_large_tAll_v10k_sl512_nep1_bspd12_dn4_graccs3_lr1e-4_wup100_wd005.yaml
```

Please refer to configuration file above to later training parameters
