#!/bin/bash

# GPU / CPU: weight, attention cache, activations
declare -a percents=(
    "0 100 0 100 0 100"
    "15 85 0 100 0 100"
    # "0 100 0 100 0 100"
    # "50 50 0 100 0 100"
    # "4 96 0 100 0 100"
)


declare -a num_gpu_batch=(
    "3"
    "8"
    "12"
)

declare -a gpu_batch_size=(
    # "32"
    "48"
    "64"
    "128"
    "256"
)

declare -a prompt_len=(
    "76"
    "256"
    "512"
)

declare -a cpu_cache_compute=(
    "true"
    # "false"
)

declare -a gen_len=(
    "32"
    "64"
    "128"
    "256"
)

for percent in "${percents[@]}"
do 
    for num_gpu_batch in "${num_gpu_batch[@]}"
    do 
        for prompt_len in "${prompt_len[@]}"
        do 
            for gen_len in "${gen_len[@]}"
            do 
                ###############
                stop_larger_batches=false
                for gpu_batch_size in "${gpu_batch_size[@]}"
                do 
                    if [ "$stop_larger_batches" = true ]; then
                        echo "Skipping GPU batch size $gpu_batch_size due to previous memory errors."
                        continue
                    fi
                    
                    for cpu_cache_compute in "${cpu_cache_compute[@]}"
                    do 
                        echo "Running model with Percent: $percent, Number of GPU batches: $num_gpu_batch, GPU-batch size: $gpu_batch_size, Prompt Length: $prompt_len, Gen Length: $gen_len."

                        if [ "$cpu_cache_compute" = true ]; then
                            echo "Running with CPU cache compute."

                            # check if the file exists before running
                            percent_arr=($percent)
                            filename="fo-v0.1-gbs$gpu_batch_size-ngbs$num_gpu_batch-prompt$prompt_len-gen$gen_len-percent-${percent_arr[0]}-${percent_arr[1]}-${percent_arr[2]}-${percent_arr[3]}-${percent_arr[4]}-${percent_arr[5]}-cpu-cache.log"

                            echo "Checking if file exists: $filename"
                            # don't run if file exists
                            if [ -f "$filename" ]; then
                                echo "File exists. Skipping."
                                continue
                            fi

                            python3 -m flexgen.flex_moe --model mistralai/Mixtral-8x7B-Instruct-v0.1 --percent $percent --gpu-batch-size $gpu_batch_size --num-gpu-batches $num_gpu_batch --prompt-len $prompt_len --gen-len $gen_len --cpu-cache-compute
                        else
                            echo "Running without CPU cache compute."

                            percent_arr=($percent)
                            filename="fo-v0.1-gbs$gpu_batch_size-ngbs$num_gpu_batch-prompt$prompt_len-gen$gen_len-percent-${percent_arr[0]}-${percent_arr[1]}-${percent_arr[2]}-${percent_arr[3]}-${percent_arr[4]}-${percent_arr[5]}-gpu-cache.log"

                            echo "Checking if file exists: logs/$filename"
                            if [ -f "logs/$filename" ]; then
                                echo "File exists. Skipping."
                                continue
                            fi

                            python3 -m flexgen.flex_moe --model mistralai/Mixtral-8x7B-Instruct-v0.1 --percent $percent --gpu-batch-size $gpu_batch_size --num-gpu-batches $num_gpu_batch --prompt-len $prompt_len --gen-len $gen_len
                        fi

                        if [ $? -ne 0 ]; then
                            echo "Error encountered with GPU batch size $gpu_batch_size. Skipping larger batch sizes."
                            stop_larger_batches=true
                        fi
                    done 
                done
            done 
            #########
        done 
    done 
done

echo "All combinations have been executed."
