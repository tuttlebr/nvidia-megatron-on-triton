---
language:
- en
library_name: nemo
datasets:
- the_pile
tags:
- text generation
- pytorch
- causal-lm
license: cc-by-4.0

---
# NeMo Megatron-GPT 20B

<style>
img {
 display: inline;
}
</style>

|[![Model architecture](https://img.shields.io/badge/Model%20Arch-Transformer%20Decoder-green)](#model-architecture)|[![Model size](https://img.shields.io/badge/Params-20B-green)](#model-architecture)|[![Language](https://img.shields.io/badge/Language-en--US-lightgrey#model-badge)](#datasets)


## Model Description

Megatron-GPT 20B is a transformer-based language model. GPT refers to a class of transformer decoder-only models similar to GPT-2 and 3 while 20B refers to the total trainable parameter count (20 Billion) [1, 2].

This model was trained with [NeMo Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html).


## Getting started

Note: You will need NVIDIA Ampere or Hopper GPUs to work with this model.

### Step 1: Install NeMo and dependencies

You will need to install NVIDIA Apex and NeMo. 

```
git clone https://github.com/ericharper/apex.git
cd apex
git checkout nm_v1.11.0
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./
```

```
pip install nemo_toolkit['nlp']==1.11.0
``` 

Alternatively, you can use NeMo Megatron training docker container with all dependencies pre-installed.

### Step 2: Launch eval server 

**Note.** The example below launches a model variant with Tensor Parallelism (TP) of 4 and Pipeline Parallelism (PP) of 1 on 4 GPUs.


```
git clone https://github.com/NVIDIA/NeMo.git 
cd NeMo/examples/nlp/language_modeling
git checkout v1.11.0
python megatron_gpt_eval.py gpt_model_file=nemo_gpt20B_bf16_tp4.nemo server=True tensor_model_parallel_size=4 trainer.devices=4
```

### Step 3: Send prompts to your model!
```python
import json
import requests

port_num = 5555
headers = {"Content-Type": "application/json"}

def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data),
                        headers=headers)
    sentences = resp.json()['sentences']
    return sentences


data = {
    "sentences": ["Tell me an interesting fact about space travel."]*1,
    "tokens_to_generate": 50,
    "temperature": 1.0,
    "add_BOS": True,
    "top_k": 0,
    "top_p": 0.9,
    "greedy": False,
    "all_probs": False,
    "repetition_penalty": 1.2,
    "min_tokens_to_generate": 2,
}

sentences = request_data(data)
print(sentences)
```


## Training Data

The model was trained on ["The Piles" dataset prepared by Eleuther.AI](https://pile.eleuther.ai/). [4]

## Evaluation results

*Zero-shot performance.* Evaluated using [LM Evaluation Test Suite from AI21](https://github.com/AI21Labs/lm-evaluation)

| ARC-Challenge	| ARC-Easy | RACE-middle | RACE-high | Winogrande | RTE | BoolQA | HellaSwag | PiQA |
| ------------- | -------- | ----------- | --------- | ---------- | --- | ------ | --------- | ---- |
| 0.4403        | 0.6141  | 0.5188       | 0.4277    | 0.659     | 0.5704 | 0.6954 | 0.721 | 0.7688 | 

## Limitations

The model was trained on the data originally crawled from the Internet. This data contains toxic language and societal biases. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts.   

## References

[1] [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[2] [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[4] [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)

## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

