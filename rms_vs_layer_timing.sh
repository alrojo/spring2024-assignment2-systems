#!/bin/bash
models=("small")
for model in "${models[@]}"; do
    echo "Running script with model: $model, rms, layer, then triton"
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device -1 --backward_pass 1 --rms_layer_triton 0
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device -1 --backward_pass 1 --rms_layer_triton 1
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device -1 --backward_pass 1 --rms_layer_triton 2
done
