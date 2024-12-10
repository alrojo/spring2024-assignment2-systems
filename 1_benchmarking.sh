#!/bin/bash
models=("small" "medium" "large" "xl" "2.7B")
for model in "${models[@]}"; do
    echo "Running script with model: $model"
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 0
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 1
done
