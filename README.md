# NVIDIA NeMo Megatron

An end-to-end framework for training and deploying LLMs with billions and trillions of parameters.

![NeMo](https://developer.nvidia.com/sites/default/files/akamai/nemo/nemo-megatron-850x480.jpg)

## What is NVIDIA NeMo Megatron?

NVIDIA NeMo Megatron, part of the NVIDIA AI platform, offers an easy, efficient, and cost-effective containerized framework to build and deploy LLMs. Designed for enterprise application development, it builds upon the most advanced technologies from NVIDIA research and provides an end-to-end workflow for automated distributed data processing, training large-scale customized GPT-3, T5, and multilingual T5 (mT5) models, and deploying models for inference at scale.

Harnessing the power of LLMs is made easy through validated and converged recipes with predefined configurations for training and inference. Customizing models is simplified by the hyperparameter tool, which automatically searches for the best hyperparameter configurations and performance for training and inference on any given distributed GPU cluster configuration.

NeMo Megatron also allows for efficiently adapting models for different use cases using prompt-based learning capabilities, such as p-tuning and prompt tuning. These methods are more efficient than traditional fine-tuning and allow LLMs to adapt to new use cases without fine-tuning the full pretrained models.

NeMo Megatron is part of NeMo, an open-source framework for building high-performance and flexible applications for conversational AI, speech AI, and biology.

## Explore the benefits

![speedometer](https://developer.nvidia.com/sites/default/files/akamai/nemo/m48-speed-256px-blk.png)

**Fastest training on GPUs.**

Use state-of-the-art (SOTA) training techniques to maximize throughput and minimize training time for LLMs with billions or trillions of parameters.

![ai](https://developer.nvidia.com/sites/default/files/akamai/nemo/m48-ai-customization-256px-blk.png)

**Validated recipes.**

Access recipes for training multiple GPT-3, T5, and mT5 models to convergence and deploy for inference

![flex](https://developer.nvidia.com/sites/default/files/akamai/nemo/m48-electronic-design-automation-256x-blk.png)

**Flexible and customizable.**

Train and deploy custom LLMs from scratch with data preprocessing, training, evaluation, and inference. Equipped with fine-tuning and prompt-based learning capabilities to customize for different use cases.

![anywhere](https://developer.nvidia.com/sites/default/files/akamai/nemo/m48-virtual-pc-cloud-computer-256x-blk.png)

**Run on prem and in the cloud.**

Train and deploy LLMs of any size on any GPU infrastructure. Supported on NVIDIA DGX SuperPOD™, NVIDIA DGX™ Foundry, Microsoft Azure, Oracle Cloud Infrastructure, and Amazon Web Services.

## Key product features

NeMo Megatron delivers high training efficiency, making large-scale natural language processing (NLP) practical, using parallelism techniques such as:

- Tensor parallelism to scale models within nodes
- Data and pipeline parallelism to scale data and models across thousands of GPUs
- Sequence parallelism to distribute activation memory across tensor parallel devices

Alongside tensor parallelism, selective activation recomputing optimizes recomputation and memory usage across tensor parallel devices during backpropagation.

It also comes equipped with fine-tuning capabilities, alongside prompt-based learning techniques, that enable customization for different datasets with minimal data, vastly improving performance and few-shot tasks.

[Read Blog](https://developer.nvidia.com/blog/nvidia-ai-platform-delivers-big-gains-for-large-language-models/)

![parallelism](https://developer.nvidia.com/sites/default/files/akamai/nemo/sota-training-techniques.jpg)

![FasterTransformer](https://developer.nvidia.com/sites/default/files/akamai/nemo/optimized-inference.jpg)

## Optimized inference.

NeMo Megatron supports deploying LLMs for inference using NVIDIA Triton™ Inference Server. With powerful optimization from Faster Transformer, you can achieve state-of-the-art accuracy, latency, and throughput inference performance on single-GPU, multi-GPU, and multi-node configurations.

NeMo Megatron makes LLMs accessible by solving many of the existing pain points across the entire stack, allowing users to easily deploy applications at scale quickly and efficiently.

[Learn more about NVIDIA Triton](https://developer.nvidia.com/nvidia-triton-inference-server)

## Comprehensive preprocessing.

NeMo Megatron allows you to bring your own dataset and tokenize data to a digestible format. It includes comprehensive preprocessing capabilities for data filtration, deduplication, blending, and formatting on datasets, on Piles and multilingual C4 (mC4). These help researchers and engineers save months of development and compute time, letting them focus on building applications.

![preprocessing](https://developer.nvidia.com/sites/default/files/akamai/nemo/comprehensive-preprocessing.jpg)

![performance](https://developer.nvidia.com/sites/default/files/akamai/nemo/recipes-and-tools.jpg)

## Easy-to-use recipes and tools.

NeMo Megatron includes prepackaged scripts, reference examples, and documentation across the entire pipeline, making LLMs possible from day one.

Several validated and converged recipes for various model sizes, for GPT-3 and T5/mT5 architectures, allow for easy training and deployment of LLMs.

Custom LLMs are also made easy through a unique offering from NeMo Megatron—the hyperparameter tool, which automatically searches for the best hyperparameter configurations to optimize training and inference for any given multi-GPU configuration, training, or deployment constraints.

## Experience large language models using NVIDIA NeMo LLM service.

![LLM-Service](https://developer.nvidia.com/sites/default/files/akamai/nemo/nemo-llm-service-630x354.jpg)

The promise of LLMs serving several use cases with a single model can enable enterprises to develop a plethora of applications, ranging from content generation to text summarization to chatbots. Yet, using customized LLMs isn’t for the faint of heart, requiring immense amounts of compute resources and technical expertise.

NVIDIA NeMo LLM Service running on the NVIDIA AI platform provides the fastest path to customizing and using LLMs for AI applications. Developers can run them in private and public clouds, run them through an API, or experiment via a playground. Models can be trained from any framework or run out of the box. State-of-the-art prompt learning capabilities, which are compute-efficient techniques, embed context in user queries to allow for greater accuracy in specific use cases.

## Getting Started

### Requirements

- Amphere or Hopper Generation GPU (>1 for Tensor Parallelism)
- Docker & Docker Compose
- Ubuntu 20.04 or newer

### Build Images

```
docker compose build
```

**nemo-gpt20b-convert**: Convert NeMo model to Triton+FasterTransformer format for tensor parallelism. The default is 2 gpus, if you have fewer or more, please read `models/triton_models/README.md` and change the `NUM_GPU` variable in the `entrypoint.sh` file.

**fastertransformer-triton-server**: Triton server with FasterTransformer backend configured to serve NVIDIA's 20B parameter Megatron GPT type LLM.

**nemo-client**: A simple JupyterLab container from which you can send requests to the Triton server and begin experimenting, locally. Start with `workspace/Evaluation.ipynb` for some examples!

### Launch Containers

**nemo-gpt20b-convert**: `docker compose up nemo-gpt20b-convert`

**fastertransformer-triton-server**: `docker compose up fastertransformer-triton-server`

**nemo-client**: `docker compose up nemo-client`

### Model Inference Tuning Parameters

| **Parameter**              | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Number of Tokens           | _Specifies how much text to generate. Tokens can be either an entire word, or parts of words. For English, 100 tokens form approximately 75 words._                                                                                                                                                                                                                                                                                                                                                    |
| Temperature                | _Controls the randomness of selecting the next token during text generation. Lower values reduce randomness, suitable for tasks with a correct answer such as question answering or summarization. Higher values increase randomness, suitable for tasks that require creativity. The [0.5, 0.8] range is a good starting point for experimentation._                                                                                                                                                  |
| Top K                      | _Controls the randomness of selecting the next token during text generation. The number of highest-probability tokens to keep, from which the next token will be selected at random. Lower values reduce randomness, suitable for tasks with a correct answer such as question answering or summarization. Higher values increase randomness, suitable for tasks that require creativity. 0 means Top K is not used. 1 means greedy decoding, that is, always selecting the most probable token next._ |
| Top P                      | _Controls the randomness of selecting the next token during text generation. This determines the minimum number of highest-probability tokens whose probabilities sum to or exceed the Top P value, from which the next token will be selected at random. Lower values reduce randomness, suitable for tasks with a correct answer such as question answering or summarization. Higher values increase randomness, suitable for tasks that require creativity._                                        |
| Repetition Penalty         | _How much to penalize tokens based on how frequently they occur in the text. A value of 1 means no penalty, while values larger than 1 discourage repeated tokens._                                                                                                                                                                                                                                                                                                                                    |
| Length Penalty             | _Only applies to beam search, that is, when the beam width is >1. Larger values penalize long candidates more heavily thus preferring shorter candidates._                                                                                                                                                                                                                                                                                                                                             |
| Beam Search Diversity Rate | _Only applies to beam search, that is, when the beam width is >1. A higher value encourages beam search to return a more diverse set of candidates._                                                                                                                                                                                                                                                                                                                                                   |
| Beam Width                 | _The number of concurrent candidates to keep track of during beam search. Higher values increase the chance of finding a good output but also require more computation. Streaming is supported with a “beam width” hyperparameter set to 1 only._                                                                                                                                                                                                                                                      |
| Random Seed                | _The model generates random results. Changing the random seed alone will produce a different response with similar characteristics. It is possible to reproduce results by fixing the random seed (assuming all other hyperparameters are also fixed)._                                                                                                                                                                                                                                                |
| Stop Words                 | _Set of character sequences, upon generating any of which, the API will stop generating any further text prematurely, even if the output length has not yet reached the specified number of tokens. It is useful to design a stopping template in the examples given to the model so that it can learn to stop appropriately upon completing an intended task._                                                                                                                                        |
