#!/bin/bash

launcher=python
command=$1
user_name=`eai user get --field name`
base_command=${command%.*}
job_config_file=$base_command.yaml

if [ ! -z "$2" ]
then
    launcher="accelerate launch"
fi

if [ -f "$job_config_file" ]; then
    job_spec="-f $job_config_file"
else 
    job_spec="--cpu 4 --mem 32"
fi


echo $launcher
echo $command
echo $base_command
echo $user_name
echo $CONDA_PREFIX
echo $job_spec

eai job new --preemptable --restartable --image $THIS_IMAGE \
    $job_spec \
    --env HOME=/home/toolkit \
    --env PYTHONPATH=`pwd` \
    --workdir `pwd` \
    --data snow.$user_name.home:/home/toolkit:rw \
    --data snow.code_llm.data:/data:rw \
    --data snow.repo_code_llm.base:/repo_data:rw \
    -- bash -c "source ~/.bashrc && conda activate $CONDA_PREFIX && $launcher $command"
