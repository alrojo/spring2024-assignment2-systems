#!/bin/bash
models=("medium")
for model in "${models[@]}"; do
    echo "Running script with model: $model"
    python cs336-systems/cs336_systems/profile_model.py --model "$model" --device "cuda:0" --backward_pass 0 --profile_memory 1
done
