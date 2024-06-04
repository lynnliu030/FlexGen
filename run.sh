#!/bin/bash

# GPU / CPU: weight, attention cache, activations
declare -a percents=(
    # "0 100 0 100 0 100"
    # "15 85 0 100 0 100"
    # "4 96 0 100 0 100"
    "50 50 0 100 0 100"
)


declare -a num_gpu_batch=(
    "3"
    # "8"
    # "12"
)

declare -a gpu_batch_size=(
    "32"
    # "48"
    # "64"
    # "128"
    # "256"
)

declare -a prompt_len=(
    "76"
    # "256"
    # "512"
)

declare -a cpu_cache_compute=(
    "true"
    # "false"
)

for percent in "${percents[@]}"
do 
    for num_gpu_batch in "${num_gpu_batch[@]}"
    do 
        stop_larger_batches=false
        for gpu_batch_size in "${gpu_batch_size[@]}"
        do 
            if [ "$stop_larger_batches" = true ]; then
                echo "Skipping GPU batch size $gpu_batch_size due to previous memory errors."
                continue
            fi

            for prompt_len in "${prompt_len[@]}"
            do 
                for cpu_cache_compute in "${cpu_cache_compute[@]}"
                do 
                    echo "Running model with Percent: $percent, Number of GPU batches: $num_gpu_batch, GPU-batch size: $gpu_batch_size, Prompt Length: $prompt_len"

                    if [ "$cpu_cache_compute" = true ]; then
                        echo "Running with CPU cache compute."
                        python3 -m flexgen.flex_moe --model mistralai/Mixtral-8x7B-Instruct-v0.1 --percent $percent --gpu-batch-size $gpu_batch_size --num-gpu-batches $num_gpu_batch --prompt-len $prompt_len --cpu-cache-compute
                    else
                        echo "Running without CPU cache compute."
                        python3 -m flexgen.flex_moe --model mistralai/Mixtral-8x7B-Instruct-v0.1 --percent $percent --gpu-batch-size $gpu_batch_size --num-gpu-batches $num_gpu_batch --prompt-len $prompt_len
                    fi

                    if [ $? -ne 0 ]; then
                        echo "Error encountered with GPU batch size $gpu_batch_size. Skipping larger batch sizes."
                        stop_larger_batches=true
                    fi
                done 
            done
        done 
    done 
done

echo "All combinations have been executed."
