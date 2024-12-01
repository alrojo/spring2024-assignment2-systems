#!/bin/bash
models=("small")
for model in "${models[@]}"; do
    echo "Running script with model: $model, testing w w/o mixed precision"
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device -1 --backward_pass 1 --mixed_precision 1
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device -1 --backward_pass 1 --mixed_precision 1
done
