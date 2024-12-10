#!/bin/bash
models=("medium")
for model in "${models[@]}"; do
    echo "Running script with model: $model"
    python cs336-systems/cs336_systems/profile_model.py --model "$model" --device "cuda" --backward_pass 0
done
