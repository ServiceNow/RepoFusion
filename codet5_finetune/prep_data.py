import json
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path

import datasets
import yaml

def params():
    '''
    A placeholder for params to be replaces with actual arg parsing later
    '''

    with open('codet5_finetune/config.yaml') as f:
        data = yaml.safe_load(f)
    
    opt = type('Opt', (), data)()
    
    return opt

def get_java_files_remove_files_present_in_reconstructed_repos(opt):
    '''
    Separates Java files from the stack 1.1 used to train SantaCoder
    and removes files used in reconstruction of a set 1K repos form the dataset
    '''
    opt.path_java_filtered = Path(opt.path_java_filtered)
    
    df = pd.read_parquet(opt.filename_1K_20plus_file_list)
    file_full_paths = set((df['name'] + '/' + df['path']).str.lower())

    opt.filename_the_stack11_dedup_alt_comments = Path(opt.filename_the_stack11_dedup_alt_comments)
    ds = datasets.load_dataset(
        str(opt.filename_the_stack11_dedup_alt_comments.parent),
        data_files=opt.filename_the_stack11_dedup_alt_comments.name,
        num_proc=opt.num_proc
    )
    ds = ds.filter(
        lambda data: (
            data['lang'] == 'Java'
            and not ((data['max_stars_repo_name'] + '/' + data['max_stars_repo_path']).lower() in file_full_paths)
        ),
        num_proc=opt.num_proc
    )

    return ds

def select_split(ds, opt):
    '''
    Selects subset on filtered data and split to train test and validation
    '''
    ds["train"] = ds["train"].shuffle(seed=opt.seed).select(
        range(opt.train_size+opt.validation_size+opt.test_size)
    )
    datasets_train_test = ds["train"].train_test_split(
        seed=opt.seed, test_size=opt.validation_size
    )
    datasets_train_validation = datasets_train_test["train"].train_test_split(
        seed=opt.seed, test_size=opt.test_size
    )

    ds["train"] = datasets_train_validation["train"]
    ds["validation"] = datasets_train_validation["test"]
    ds["test"] = datasets_train_test["test"]
    
    return ds


def process(opt):
    ds = get_java_files_remove_files_present_in_reconstructed_repos(opt)
    ds = select_split(ds, opt)
    ds.save_to_disk(opt.path_java_filtered_subset, num_proc=opt.num_proc)

if __name__ == '__main__':
    opt = params()
    process(opt)
    
    
    
        
