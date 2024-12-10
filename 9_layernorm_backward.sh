#!/bin/bash
python cs336-systems/cs336_systems/layernorm.py --backward 1
models=("medium")
for model in "${models[@]}"; do
    echo "Running script with model: $model, rms, layer, then triton"
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 1 --norm_layer_type rms
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 1 --norm_layer_type layer
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 1 --norm_layer_type triton
    python cs336-systems/cs336_systems/benchmark_model.py --model "$model" --device "cuda" --backward_pass 1 --norm_layer_type compile_rms
done
