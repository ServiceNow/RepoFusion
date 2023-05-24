## Intro
This folder contais code to prepare dataset and fine tune `codet5` models on the next token prediction (NTP) objective on the Java subset of `The Stack`. Please see the paper for details on the dataset. 

## Dataset preparation
Modify the cofing file `[this repo root]/Finetuning_CodeT5/config/base.yaml`. Only parameters under `Data preparation params` has effect on data preparation. Set the following parmeter to local target dataset path:
- `path_java_filtered_subset_root` - destination path for the fine tuning dataset
- the rest of the parameters control path to a list of files to exclude, source dataseet name, split sizes, seed etc.

Set `PYTHONPATH` to the this repository root. From this repository root run:
```
python Finetuning_CodeT5/prep_data.py
```

## Finetune codet5 model on prepared data
Assuming the previous step is done, modify the following parameter in the cofing file `[this repo root]/Finetuning_CodeT5/config/base.yaml`:
- `model_dir_base` - the distination path base folder for checkpoints and tensorboard to be stored

In our experiments, we used machine with 2 A100-80GB for finetuning base model and with 4 for large. If you are working with a different hardware setting, please modify the `per_device_train_batch_size` and `gradient_accumulation_steps` in `[this repo root]/Finetuning_CodeT5/config/finetuned_codet5base_512.yaml` and `[this repo root]/Finetuning_CodeT5/config/finetuned_codet5large_512.yaml` according to your gpu capacity and count, other parameters like learinig rate or weight decay can be affected as well.

Set `PYTHONPATH` to the this repository root

From this repository root run the following command to finetune and evaluate the base model:
```
accelerate launch --multi_gpu --num_processes=2 Finetuning_CodeT5/train.py Finetuning_CodeT5/config/finetuned_codet5base_512.yaml
```

From this repository root run the following command to finetune the base model:
```
accelerate launch --multi_gpu --num_processes=4 Finetuning_CodeT5/train.py Finetuning_CodeT5/config/finetuned_codet5large_512.yaml
```

Please refer to configuration file above to later training parameters
