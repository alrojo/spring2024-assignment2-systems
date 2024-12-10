#!/bin/bash
models=("medium")
for model in "${models[@]}"; do
    echo "Running script with model: $model, rms, layer, then triton"
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 0 --norm_layer_type rms
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 0 --norm_layer_type layer
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 0 --norm_layer_type triton
done
