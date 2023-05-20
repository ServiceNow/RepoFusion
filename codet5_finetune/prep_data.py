import json
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import numpy.random
import os

import datasets
import yaml
import random

from transformers import AutoTokenizer

from codet5_finetune.options import options
from codet5_finetune.util import set_global_seeds

# Hack from here: https://github.com/huggingface/datasets/issues/1785#issuecomment-1305872400
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


def get_java_files_remove_files_present_in_reconstructed_repos(opt):
    """
    Separates Java files from the stack 1.1 used to train SantaCoder
    and removes files used in reconstruction of a set 1K repos form the dataset
    used for Stack Repo.
    """
    df = pd.read_parquet(opt.filename_1K_20plus_file_list)
    used_repo_names = set(df["name"].str.lower())

    opt.filename_the_stack11_dedup_alt_comments = Path(
        opt.filename_the_stack11_dedup_alt_comments
    )
    ds = datasets.load_dataset(
        str(opt.filename_the_stack11_dedup_alt_comments.parent),
        data_files=opt.filename_the_stack11_dedup_alt_comments.name,
        num_proc=opt.num_proc,
    )
    ds = ds.filter(
        lambda data: (
            data["lang"] == "Java"
            and not (data["max_stars_repo_name"].lower() in used_repo_names)
        ),
        num_proc=opt.num_proc,
    )

    return ds


def get_repo_names_rand_seq(ds, opt):
    repo_names_rand_seq_filename = (
        opt.path_java_filtered_subset_root / opt.repo_names_rand_seq_filename
    )
    if repo_names_rand_seq_filename.is_file() and not opt.regenerate:
        return json.loads(repo_names_rand_seq_filename.read_text())
    repo_names = list(ds["train"]["max_stars_repo_name"])
    repo_names_dict = defaultdict(int)
    for el in repo_names:
        repo_names_dict[el] += 1
    repo_names_20plus = []
    for k, v in repo_names_dict.items():
        if v >= opt.min_file_count_per_repo:
            repo_names_20plus.append((k, v))

    random.seed(opt.seed)
    random.shuffle(repo_names_20plus)
    repo_names_rand_seq_filename.write_text(json.dumps(repo_names_20plus))
    return repo_names_20plus


def get_repo_names_for_pivots_size(repo_names, pivots_count, opt):
    cnt = 0
    names = []
    for el in repo_names:
        cnt_4_repo = el[1] * opt.pivots_per_file
        if cnt_4_repo > opt.max_pivots_per_repository:
            cnt_4_repo = opt.max_pivots_per_repository
        names.append(el)
        if pivots_count != -1 and (cnt + cnt_4_repo >= pivots_count):
            break
        cnt += cnt_4_repo
    return names


def get_split_repo_names(repo_names, opt):
    splits_filename = opt.path_java_filtered_subset_root / opt.splits_filename
    if splits_filename.is_file() and not opt.regenerate:
        return json.loads(splits_filename.read_text())
    validation_names = get_repo_names_for_pivots_size(
        repo_names, opt.validation_size, opt
    )
    repo_names = repo_names[len(validation_names) :]
    test_names = get_repo_names_for_pivots_size(repo_names, opt.test_size, opt)
    repo_names = repo_names[len(test_names) :]
    train_names = get_repo_names_for_pivots_size(repo_names, opt.train_size, opt)
    split_repo_names = {
        "train": train_names,
        "validation": validation_names,
        "test": test_names,
    }

    splits_filename.write_text(json.dumps(split_repo_names))
    return split_repo_names


def get_sub_slit(ds, split_rep_names, opt):
    names = set(el[0] for el in split_rep_names)
    return ds.filter(
        lambda data: (
            data["lang"] == "Java" and (data["max_stars_repo_name"] in names)
        ),
        num_proc=opt.num_proc,
    )["train"]


def get_ds_subs(ds, split_rep_names, opt):
    ds_out = datasets.DatasetDict()
    for k, v in split_rep_names.items():
        ds_out[k] = get_sub_slit(ds, v, opt)
    return ds_out


def is_data_generated(opt):
    try:
        ds = datasets.load_from_disk(
            opt.path_java_filtered_subset_root / opt.java_filtered_subset_data_dir
        )
        return True
    except Exception:
        return False


def prepare_data_subset(opt):
    """
    Prepares data by taking filtered and deduplicated subset of The Stack 1.1
    which was used to traind SantaCoder model, takes java files which have
    not been used in recreation of the 1K repositories for repo level context
    experiments
    """
    if is_data_generated(opt) and not opt.regenerate:
        return

    ds = get_java_files_remove_files_present_in_reconstructed_repos(opt)

    repo_names = get_repo_names_rand_seq(ds, opt)
    split_repo_names = get_split_repo_names(repo_names, opt)

    ds_out = get_ds_subs(ds, split_repo_names, opt)
    ds_out.save_to_disk(
        opt.path_java_filtered_subset_root / opt.java_filtered_subset_data_dir,
        num_proc=opt.num_proc,
    )


def get_pivots_for_split(split, ds, opt, rand_generators, tokenizer):
    is_random = split == "train"
    if is_random:
        # pivot: None means it will be randomly generated during collate
        def gen_func():
            for idx in range(len(ds)):
                for _ in range(opt.pivots_per_file):
                    yield {"data_idx": idx, "pivot": None, "split": split}

        return datasets.Dataset.from_generator(gen_func)
    else:

        def process_sample(sample, idx, rank):
            rnd = rand_generators[rank]
            sample_tokens_count = len(tokenizer(sample["content"][0])["input_ids"])
            return {
                "data_idx": [idx[0]] * opt.pivots_per_file,
                "pivot": list(
                    rnd.choice(sample_tokens_count, opt.pivots_per_file, replace=False)
                ),
                "split": [split] * opt.pivots_per_file,
            }

        return ds.map(
            process_sample,
            with_indices=True,
            with_rank=True,
            remove_columns=ds.column_names,
            num_proc=opt.num_proc,
            batched=True,
            batch_size=1,
        )


def prepare_pivots_data(opt):
    ds = datasets.load_from_disk(
        opt.path_java_filtered_subset_root / opt.java_filtered_subset_data_dir
    )
    ds_pivots = datasets.DatasetDict()
    rand_generators = [np.random.default_rng(opt.seed + i) for i in range(opt.num_proc)]
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model_name)
    for k, v in ds.items():
        ds_pivots[k] = get_pivots_for_split(k, v, opt, rand_generators, tokenizer)
    ds_pivots.save_to_disk(
        opt.path_java_filtered_subset_root / opt.java_filtered_subset_pivots_dir,
        num_proc=opt.num_proc,
    )


def run(opt):
    opt.path_java_filtered_subset_root = Path(opt.path_java_filtered_subset_root)
    opt.path_java_filtered_subset_root.mkdir(parents=True, exist_ok=True)
    prepare_data_subset(opt)
    prepare_pivots_data(opt)


if __name__ == "__main__":
    opt = options()
    set_global_seeds(opt)
    run(opt)
