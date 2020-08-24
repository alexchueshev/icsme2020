#!/bin/bash

command="$1"

export PYTHONPATH=$(pwd)/src
export OPENBLAS_NUM_THREADS=1

echo Command: $command

source ./venv/bin/activate

"$(pwd)/venv/bin/python" "$(pwd)/main.py"  \
    -i "$(pwd)/example/kafka.yaml" \
     -prj kafka \
     "$command"

deactivate
