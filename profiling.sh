#!/bin/bash
models=("small")
for model in "${models[@]}"; do
    echo "Running script with model: $model"
    python cs336-systems/cs336_systems/profile_model.py --model "$model" --device -1 --backward_pass 1
done
