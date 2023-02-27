# repo_training_codellm

## Install

instal pytotch seapratelly for example 
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

Then install the rest,with remved torch, for some reaosn does not work in one command:
```
conda install  --channel=conda-forge --file FiD/requirements.txt
```

Also needed to uninstall tokenizer and transformers from conda foerge and instal with pip to get the latest veriosn, tokenizer 0.13.2, otherwise had libssl problems or tokenizer compatibility problems

Install jupyter lab


Install `pip install ConfigArgParse SentencePiece`
 

## Run lab session in research-interactive-toolkit
```
export PYTHONPATH=/home/toolkit/code/
jupyter lab --ip=0.0.0.0 --port=8080 --no-browser     \
    --notebook-dir=/     --NotebookApp.token=''     \
    --NotebookApp.custom_display_url=https://${EAI_JOB_ID}.job.console.elementai.com    \
    --NotebookApp.disable_check_xsrf=True     --NotebookApp.allow_origin='*'
```

