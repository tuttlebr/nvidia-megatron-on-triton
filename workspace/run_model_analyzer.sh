#!/bin/bash
clear
pip3 install -q --upgrade triton-model-analyzer
rm -rf /workspace/model_analyzer_results
mkdir -p /workspace/model_analyzer_results

model-analyzer profile -f /workspace/sweep.yaml