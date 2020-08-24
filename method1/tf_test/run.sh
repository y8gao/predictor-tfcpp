#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

MODEL_PATH=$1
if [[ ! (-f $MODEL_PATH) ]]; then
    echo "Error: No such model: $MODEL_PATH"
    exit 2
fi

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:3rd-party/tensorflow/bazel-bin/tensorflow/ ./tf_test $MODEL_PATH

