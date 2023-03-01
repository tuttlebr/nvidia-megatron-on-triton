Will be populated after container start.

Modify the following file based on how you converted.

`models/triton_models/gpt/fastertransformer/config.pbtxt`
If you used 2 GPUs during the model conversion...

```bash
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "2"
  }
}
```

```bash
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/models/custom/custom-nemo_gpt20B/2-gpu/"
  }
}
```
