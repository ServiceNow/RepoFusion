#!/bin/bash

if [[ $1 == "-a" ]]; then
    use_accelerate=1
    shift

elif [[ $1 == "-f" ]]; then
    run_fid=1
    shift
fi

launcher=python
command=$1
eai_user_name=`eai account get --field name`
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

prog_workdir=`pwd`
prog_pythonpath=$prog_workdir

if [ ! -z ${use_accelerate+x} ]; then
    launcher="accelerate launch --multi_gpu --num_processes=$num_gpu"
fi

if [ ! -z ${run_fid+x} ]; then
    launcher="torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpu"
    prog_pythonpath=$prog_workdir/FiD
fi

echo $launcher
echo $command
echo $base_command
echo $eai_user_name
echo $CONDA_PREFIX
echo $job_spec
echo $num_gpu
echo $launcher_args
echo $prog_workdir
echo $prog_pythonpath

eai job new --preemptable --restartable --image $THIS_IMAGE \
    $job_spec \
    --env HOME=/home/toolkit \
    --env PYTHONPATH=$prog_pythonpath \
    --workdir $prog_workdir \
    --data snow.$eai_user_name.home:/home/toolkit:rw \
    --data snow.code_llm.data:/data:ro \
    --data snow.repo_code_llm.base:/repo_data:rw \
    -- bash -c "source ~/.bashrc && conda activate $CONDA_PREFIX && $launcher $launcher_args"
