#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=2
N_CORES_PER_GPU=16

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=mtbench_run_moe_dist.py

pgrep -fl python | awk '!/mtbench_run_moe_dist\.py/{print $1}' | xargs sudo kill

set -x

echo "MY_IPADDR: $MY_IPADDR"

mpirun \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --gbs 1 \
    --num-gb 4 \
    --gen-len 32 \
    --percent 1 99 0 100 0 100 \
    --comm-device cpu \
    --cpu-cache-compute \
    --no-log >> moe_dist_gen32_1_99_0_100_0_100_1_4_cpu.log 2>&1