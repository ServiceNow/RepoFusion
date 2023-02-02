This directory contains code and instructions in support of our paper [RepoFusion: Training Code Models to Understand Your Repository](https://arxiv.org/abs/2306.10998). For more details, please see the paper.

## Dependencies

- Python >= 3.7
- [PyTorch](http://pytorch.org/) 
- [Transformers](http://huggingface.co/transformers/)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [Datasets](https://huggingface.co/docs/datasets/index)
- [NumPy](http://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [tqdm](https://tqdm.github.io/)

# Data

### Download data
We create and release Stack-Repo, a dataset of 200 Java repositories from GitHub with permissive licenses and near-deduplicated files that are augmented with three types of repository contexts.

- Prompt Proposal (PP) Contexts: These contexts are based on the prompt proposals from the paper [Repository-Level Prompt Generation for Large Language Models of Code] (https://arxiv.org/abs/2206.12839).
    
- BM25 Contexts: These contexts are obtained based on the BM25 similarity scores.
    
- RandomNN Contexts: These contexts are obtained using the nearest neighbors in the representation space of an embedding model.It contains three folders corresponding to our train, validation, and test splits. Each split contains a separate folder for a repository where each repository contains all .java files in the repository in the original directory structure along with three .json files corresponding to the PP, BM25, and RandomNN repo contexts.  

For more details of the dataset including the usage and instructions to download, please see [here](https://huggingface.co/datasets/RepoFusion/Stack-Repo). 

# Trained Checkpoints
The trained checkpoints can be downloaded using [here](https://huggingface.co/RepoFusion/trained_checkpoints). We have released the following checkpoints:

- `RepoFusion_PPC`: RepoFusion model trained with prompt proposal repo contexts. This is our best-performing model.

- `RepoFusion_BM25`: RepoFusion model trained with BM25 repo contexts.

- `RepoFusion_RandomNN`: RepoFusion model trained with RandomNN repo contexts.

- `finetuned_codet5base_512`: Our finetuned CodeT5-base model. This was used as initialization for our RepoFusion models.
  
- `finetuned_codet5large_512`: Our finetuned CodeT5-large model. This was used as a baseline.

# Implementation of RepoFusion 

RepoFusion can be trained using [`train_reader.py`](train_reader.py) and evaluated with [`test_reader.py`](test_reader.py). Please see RepoFusion/src/options.py for a complete list of arguments.

### Train

[`train_reader.py`](train_reader.py) provides the code to train a model. An example usage of the script is given below:

```shell
torchrun --standalone --nnodes=1 --nproc_per_node=1 RepoFusion/train_reader.py \
        --dataset_path=../the-stack-repo/ \
        --data_file_pattern=*/hole_and_PP_contexts.json \
        --n_context=32 \ #number of repo contexts, $N$ in the paper
        --text_maxlength=768 \  #max length of repo context, $l$ in the paper
        --scheduler=linear \
        --lr=1e-5 \
        --save_freq=5000 \
        --initialize_from_pretrained \ # whether to initialize from a pretrained model or finetuned model
        --finetuned_model_path=../trained_checkpoints/finetuned_codet5base_512 \
        --passage_mode=no-truncation-codex-last \ # type of repo context creation and ordering starategy to use. no-truncation-codex-last corresponds to NT-Prior-Last in the paper.
        --checkpoint_dir=../checkpoints/ \
        --name=PPC_768_32\
```

### Test

You can evaluate your model or a pretrained model with [`test_reader.py`](test_reader.py). An example usage of the script for evaluating the a trained RepoFusion model is provided below.

```shell
torchrun --standalone --nnodes=1 --nproc_per_node=1 RepoFusion/test_reader.py \
        --dataset_path=../the-stack-repo/ \
        --eval_split_name=test \
        --output_dir=../../outputs/ \
        --passage_mode=no-truncation-codex-last \
        --trained_model_path=../trained_checkpoints/RepoFusion_PPC \
        --n_context=32 \
        --text_maxlength=768 \
        --per_gpu_batch_size 1 \
        --num_of_eval_examples_per_gpu=-1 \ # -1 for all examples
        --name=PPC_test_768_32 
```
To evaluate with a model other than CodeT5-base, add arguments for model_name, model_size and model_type. For example, to evalaute CodeGen-2B-multi, add `--model_name=Salesforce/codegen-2B --model_size=multi --model_type=codegen` to the above command.

In addition, use `passage_mode=toprule+prior` for post+prior experiments and `passage_mode=pretrained` for prior experiments.
When evaluating on CodeT5 finetuned models, use `passage_mode=finetuned`. In all these cases, do not provide the trained_model_path argument. 

The predictions from the models along with the ground truth can be found in the outputs directory. Use `calculate_metrics.py` to calculate the completion success rates.

# Finetuning CodeT5

For details on the implementation of finetuning CodeT5 models, please refer to the README file inside the Finetuning_CodeT5 directory.

# References

Most of our implementation is built on top of the following sources:

[1] G. Izacard, E. Grave [*Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*](https://arxiv.org/abs/2007.01282)

```bibtex
@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
      author={Gautier Izacard and Edouard Grave},
      url = {https://arxiv.org/abs/2007.0128},
      year={2020},
      publisher = {arXiv},
}
```
[2] D. Shrivastava, H. Larochelle, D. Tarlow [*Repository-Level Prompt Generation for Large Language Models of Code*](https://arxiv.org/abs/2206.12839)

```bibtex
@article{shrivastava2022repository,
  title={Repository-level prompt generation for large language models of code},
  author={Shrivastava, Disha and Larochelle, Hugo and Tarlow, Daniel},
  journal={arXiv preprint arXiv:2206.12839},
  year={2022}
}
```

# Citation
If you use our data, trained checkpoints or code, please cite us as below:
```
@article{shrivastava2023repofusion,
  title={RepoFusion: Training Code Models to Understand Your Repository},
  author={Shrivastava, Disha and Kocetkov, Denis and de Vries, Harm and Bahdanau, Dzmitry and Scholak, Torsten},
  journal={arXiv preprint arXiv:2306.10998},
  year={2023}
}

```
# License

See the [LICENSE](LICENSE) file for more details.
