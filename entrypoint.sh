#!/bin/bash

NUM_GPU=2
export PYTHONPATH=/workspace/FasterTransformer:${PYTHONPATH}

rm -rf /models/custom/custom-nemo_gpt20B/${NUM_GPU}-gpu

python3 FasterTransformer/examples/pytorch/gpt/utils/nemo_ckpt_convert.py \
        --in-file /models/nemo-megatron-gpt-20B/nemo_gpt20B_bf16_tp4.nemo \
        --infer-gpu-num ${NUM_GPU} \
        --saved-dir /models/custom/custom-nemo_gpt20B \
        --weight-data-type fp16 \
        --load-checkpoints-to-cpu 0

cp -r /workspace/fastertransformer_backend/all_models/gpt \
        /models/triton_models/

chmod 777 /models/triton_models/gpt/fastertransformer/config.pbtxt


cat <<EOF
Modify the following file based on how you converted.

Edit this file: models/triton_models/gpt/fastertransformer/config.pbtxt

For example, if you used 2 GPUs during the model conversion.

parameters {
  key: "tensor_para_size"
  value: {
    string_value: "2"
  }
}

parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/models/custom/custom-nemo_gpt20B/2-gpu/"
  }
}
EOF