#!/bin/bash

if [[ $1 == "-a" ]]; then
    use_accelerate=1
    shift
fi

launcher=python
command=$1
eai_user_name=`eai user get --field name`
base_command=${command%.*}
job_config_file=$base_command.yaml
launcher_args="$@"

if [ -f "$job_config_file" ]; then
    job_spec="-f $job_config_file"
    num_gpu=`yq -r .resources.gpu  $job_config_file`
else 
    job_spec="--cpu 4 --mem 32"
    num_gpu=0
fi

if [ ! -z ${use_accelerate+x} ]; then
    launcher="accelerate launch --multi_gpu --num_processes=$num_gpu"
fi

echo $launcher
echo $command
echo $base_command
echo $eai_user_name
echo $CONDA_PREFIX
echo $job_spec
echo $num_processes
echo $launcher_args

eai job new --preemptable --restartable --image $THIS_IMAGE \
    $job_spec \
    --env HOME=/home/toolkit \
    --env PYTHONPATH=`pwd` \
    --workdir `pwd` \
    --data snow.$eai_user_name.home:/home/toolkit:rw \
    --data snow.code_llm.data:/data:rw \
    --data snow.repo_code_llm.base:/repo_data:rw \
    -- bash -c "source ~/.bashrc && conda activate $CONDA_PREFIX && $launcher $launcher_args"
