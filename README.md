# ScaleLLM: An efficient LLM Inference solution
[![build and test](https://github.com/vectorch-ai/ScaleLLM/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/vectorch-ai/ScaleLLM/actions/workflows/build.yml) [![GitHub Repo stars](https://img.shields.io/github/stars/vectorch-ai/ScaleLLM?style=social)](https://github.com/vectorch-ai/ScaleLLM/stargazers)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Discord](https://dcbadge.vercel.app/api/server/PKe5gvBZfn)](https://discord.gg/PKe5gvBZfn)


> **Warning**<br />
> ScaleLLM is currently in the active development stage and may not yet provide the optimal level of inference efficiency. We are fully dedicated to continuously enhancing its efficiency while also adding more features.


In the coming weeks, we have exciting plans to focus on [**_speculative decoding_**](https://github.com/orgs/vectorch-ai/projects/1) and [**_stateful conversation_**](https://github.com/orgs/vectorch-ai/projects/2), alongside further kernel optimizations. We appreciate your understanding and look forward to delivering an even better solution.


## Latest News:
* [11/2023] - First [official release](https://github.com/vectorch-ai/ScaleLLM/releases/tag/v0.0.1) with support for popular open-source models.


## Table of contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Get Started](#get-started)
  - [ScaleLLM server](#scalellm-server)
  - [Rest API Server](#rest-api-server)
  - [Chatbot UI](#chatbot-ui)
  - [Docker Compose](#docker-compose)
- [Usage Examples](#usage-examples)
- [Quantization](#quantization)
- [Limitations](#limitations)
- [Contributing](#Contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Overview

ScaleLLM is a cutting-edge inference system engineered for large language models (LLMs), meticulously designed to meet the demands of production environments. It extends its support to a wide range of popular open-source models, including Llama2, Bloom, GPT-NeoX, and more. 

## Key Features

- [High Efficiency](): Excels in high-performance LLM inference, leveraging state-of-the-art techniques and technologies like [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Paged Attention](https://github.com/vllm-project/vllm), [Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference), and more.
- [Tensor Parallelism](): Utilizes tensor parallelism for efficient model execution.
- [OpenAI-compatible API](): An efficient [golang](https://en.wikipedia.org/wiki/Go_(programming_language)) rest api server that compatible with OpenAI.
- [Huggingface models](): Seamless integration with most popular [HF models](#supported-models), supporting safetensors.
- [Customizable](): Offers flexibility for customization to meet your specific needs, and provides an easy way to add new models.
- [Production Ready](): Engineered with production environments in mind, ScaleLLM is equipped with robust system monitoring and management features to ensure a seamless deployment experience.


## Supported Models

Please note that in order to use Yi models, you need to add `--model_type=Yi` to the command line. For example:
```bash
docker pull docker.io/vectorchai/scalellm:latest
docker run -it --gpus=all --net=host --shm-size=1g \
  -v $HOME/.cache/huggingface/hub:/models \
  -e HF_MODEL_ID=01-ai/Yi-34B-Chat-4bits \
  -e DEVICE=auto \
  docker.io/vectorchai/scalellm:latest --logtostderr --model_type=Yi
```

|   Models   | Tensor Parallel | Quantization | Chat API | HF models examples |
| :--------: | :-------------: | :----------: | :------: | :---------------------------:|
|   Aquila   |       Yes       |     Yes      |    Yes   | [BAAI/Aquila-7B](https://huggingface.co/BAAI/Aquila-7B), [BAAI/AquilaChat-7B](https://huggingface.co/BAAI/AquilaChat-7B) |
|   Bloom    |       Yes       |     Yes      |    No    | [bigscience/bloom](https://huggingface.co/bigscience/bloom) |
|   Baichuan |       Yes       |     Yes      |    Yes   | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |
|   ChatGLM3 |       Yes       |     Yes      |    Yes   | [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) |
|   GPT_j    |       Yes       |     Yes      |    No    | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) |
|  GPT_NeoX  |       Yes       |     Yes      |    No    | [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) |
|    GPT2    |       Yes       |     Yes      |    No    | [gpt2](https://huggingface.co/gpt2)|
| InternLM   |       Yes       |     Yes      |    Yes   | [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) |
|   Llama2   |       Yes       |     Yes      |    Yes   | [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), [TheBloke/Llama-2-13B-chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ), [TheBloke/Llama-2-70B-AWQ](https://huggingface.co/TheBloke/Llama-2-70B-AWQ) |
|  Mistral   |       Yes       |     Yes      |    Yes   | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
|    MPT     |       Yes       |     Yes      |    Yes   | [mosaicml/mpt-30b](https://huggingface.co/mosaicml/mpt-30b) |
|   Phi2     |       Yes       |     Yes      |    No   | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) |
|   Qwen     |       Yes       |     Yes      |    Yes   | [Qwen/Qwen-72B-Chat](https://huggingface.co/Qwen/Qwen-72B-Chat) |
|    Yi      |       Yes       |     Yes      |    Yes    |[01-ai/Yi-6B](https://huggingface.co/01-ai/Yi-6B), [01-ai/Yi-34B-Chat-4bits](https://huggingface.co/01-ai/Yi-34B-Chat-4bits), [01-ai/Yi-6B-200K](https://huggingface.co/01-ai/Yi-6B-200K) |

If your model is not included in the supported list, we are more than willing to assist you. Please feel free to create a request for adding a new model on [GitHub Issues](https://github.com/vectorch-ai/ScaleLLM/issues).

## Getting Started

The easiest way to get started with our project is by using the official Docker images. If you don't have Docker installed, please follow the installation instructions for your platform. 
Below, you will find a list of all available Docker images for our project:
|   Docker Image   | cuda 12.1 | cuda 11.8 |
| :--------------: | :-------: | :-------: |
| [scalellm](https://hub.docker.com/r/vectorchai/scalellm/tags) |     Yes   | No |
| [scalellm_cu118](https://hub.docker.com/r/vectorchai/scalellm_cu118/tags) |     No   | Yes |
| [scalellm-gateway](https://hub.docker.com/r/vectorchai/scalellm-gateway/tags) |     -   | - |
| [chatbot-ui](https://hub.docker.com/r/vectorchai/chatbot-ui/tags) |     -   | - |

### Docker Installation

You can download and install Docker from the official website: [Docker Installation](https://docs.docker.com/get-docker/). To use GPUs in docker, you also need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### ScaleLLM server

Once you have Docker installed, you can run ScaleLLM Docker container with [latest image](https://hub.docker.com/r/vectorchai/scalellm/tags) using the following command:

```bash
docker pull docker.io/vectorchai/scalellm:latest
docker run -it --gpus=all --net=host --shm-size=1g \
  -v $HOME/.cache/huggingface/hub:/models \
  -e HF_MODEL_ID=TheBloke/Llama-2-7B-chat-AWQ \
  -e DEVICE=cuda:0 \
  docker.io/vectorchai/scalellm:latest --logtostderr
``` 

This command starts the Docker container with GPU support and various configuration options.

- `HF_MODEL_ID` specifies which Hugging Face model you want to run.
- `HF_MODEL_REVISION` specifies which Hugging Face model revision you want to run. By default, it is set to `"main"`.
- `DEVICE` specifies the device on which this model should run. By default, it is set to `"auto"`, using all available GPUs. You can also specify specific GPUs by using `"cuda:0,cuda:1"`, or use CPU by using `"cpu"`.
- `HF_MODEL_ALLOW_PATTERN` specifies which types of files are allowed to be downloaded. By default, it will be configured automatically based on tensor type. Only use this option if the default configuration is not working for you.
- `HUGGING_FACE_HUB_TOKEN` specifies the token from [huggingface](https://huggingface.co/settings/tokens) for gated models.

> **Warning**<br />
> * The docker image with tag '[latest](https://hub.docker.com/r/vectorchai/scalellm/tags)' could be changed to a new version upon new release. In order to use latest image, you may need to repull the image with specific tag.
> * Two version of docker images are provided for cuda 12.1 and cuda 11.8. Please choose the right image for your environment.
> * NCCL might fall back to using the host memory if NVLink or PCI is not available. To allow NCCL to use the host memory, we added '--shm-size=1g' to the docker run command.
> * Although ScaleLLM supports both `CPU` and `GPU`, we recommend using GPU for better performance. CPU support is mainly for debugging and testing purposes, so the performance might be sub-optimal. 

#### Ports and Endpoints

After running the Docker container, two ports are exposed:

1. **Port 8888 for gRPC Server**:

   The gRPC server is served on 0.0.0.0:8888 by default. You can use gRPC to interact with the service.

2. **Port 9999 for HTTP Server**:

   The simple HTTP server for instrument will be served on 0.0.0.0:9999 by default. This server provides various endpoints for managing and monitoring the service:

   - Use `curl localhost:9999/health` to check the health status of the service.
   - Use `curl localhost:9999/metrics` to export Prometheus metrics.
   - Use `curl localhost:9999/gflags` to list all available gflags for configuration.
   - add more to come...

### Rest API Server

You can also start a REST API gateway with [latest image](https://hub.docker.com/r/vectorchai/scalellm-gateway/tags) using the following command:

```bash
docker pull docker.io/vectorchai/scalellm-gateway:latest
docker run -it --net=host \
  docker.io/vectorchai/scalellm-gateway:latest --logtostderr
```

The REST API Server is available on `localhost:8080`. You can use REST API requests to interact with the system. Check out the [Usage Examples](#usage-examples) section for more details.

### Chatbot UI

A local Chatbot UI is also available on [localhost:3000](localhost:3000). You can start it with [latest image](https://hub.docker.com/r/vectorchai/chatbot-ui/tags) using the following command:

```bash
docker pull docker.io/vectorchai/chatbot-ui:latest
docker run -it --net=host \
  -e OPENAI_API_HOST=http://127.0.0.1:8080 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  docker.io/vectorchai/chatbot-ui:latest
```

### Docker Compose

Using Docker Compose is the easiest way to run ScaleLLM with all the services together. If you don't have Docker Compose installed, please follow the [installation doc](https://docs.docker.com/compose/install/) for your platform.

```bash
curl https://raw.githubusercontent.com/vectorch-ai/ScaleLLM/main/scalellm.yml -sSf > scalellm_compose.yml
HF_MODEL_ID=TheBloke/Llama-2-7B-chat-AWQ DEVICE=cuda docker compose -f ./scalellm_compose.yml up
```

you will get following running services:
* Chatbot UI on port 3000: [localhost:3000](localhost:3000)
* ScaleLLM gRPC server on port 8888: `localhost:8888`
* ScaleLLM HTTP server for monitoring on port 9999: `localhost:9999`
* ScaleLLM REST API server on port 8080: `localhost:8080`

## Usage Examples
### Chat Completions

You can get chat completions with the following example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Llama-2-7B-chat-AWQ",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

```python
import os
import sys
import openai

openai.api_base = "http://localhost:8080/v1"

# List available models
print("==== Available models ====")
models = openai.Model.list()

model = "TheBloke/Llama-2-7B-chat-AWQ"

completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    max_tokens=256,
    stream=True,
)

print(f"==== Model: {model} ====")
for chunk in completion:
    content = chunk["choices"][0]["delta"].get("content")
    if content:
        print(content, end="")
``` 

### Completions

For regular completions, you can use this example:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Llama-2-7B-chat-AWQ",
    "prompt": "hello",
    "max_tokens": 32,
    "temperature": 0.7,
    "stream": true
  }'
```

```python
import os
import sys
import openai

openai.api_base = "http://localhost:8080/v1"

# List available models
print("==== Available models ====")
models = openai.Model.list()

model = "TheBloke/Llama-2-7B-chat-AWQ"

completion = openai.Completion.create(
    model=model,
    prompt="hello",
    max_tokens=256,
    temperature=0.7,
    stream=True,
)

print(f"==== Model: {model} ====")
for chunk in completion:
    content = chunk["choices"][0].get("text")
    if content:
        print(content, end="")
```

## Quantization
Quantization is a crucial process for reducing the memory footprint of models. ScaleLLM offers support for two quantization techniques: Accurate Post-Training Quantization ([GPTQ](https://arxiv.org/abs/2210.17323)) and Activation-aware Weight Quantization ([AWQ](https://arxiv.org/abs/2306.00978)), with seamless integration into the following libraries: autogptq, exllama, exllamav2, and awq. 

>By default, **exllamav2** is employed for GPTQ 4-bit quantization. However, you have the flexibility to choose a specific implementation by configuring the **"--qlinear_gptq_impl"** option, which allows you to select from exllama, exllamav2, or auto option.

## Limitations

There are several known limitations we are looking to address in the coming months, including:

- Only supports GPUs that newer than Turing architecture.

## Contributing

If you have any questions or want to contribute, please don't hesitate to ask in our ["Discussions" forum](https://github.com/vectorch-ai/ScaleLLM/discussions) or join our ["Discord" chat room](https://discord.gg/PKe5gvBZfn). We welcome your input and contributions to make ScaleLLM even better. Please follow the [Contributing.md](https://github.com/vectorch-ai/ScaleLLM/blob/main/CONTRIBUTING.md) to get started.

## Acknowledgements
The following open-source projects have been used in this project, either in their original form or modified to meet our needs:
* [flashinfer](https://github.com/flashinfer-ai/flashinfer)
* [pytorch](https://github.com/pytorch/pytorch)
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
* [vllm](https://github.com/vllm-project/vllm)
* [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
* [llm-awq](https://github.com/mit-han-lab/llm-awq)
* [flash-attn](https://github.com/Dao-AILab/flash-attention)
* [exllama](https://github.com/turboderp/exllamav2)
* [tokenizers](https://github.com/huggingface/tokenizers)
* [safetensors](https://github.com/huggingface/safetensors/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [grpc-gateway](https://github.com/grpc-ecosystem/grpc-gateway)
* [chatbot-ui](https://github.com/mckaywrigley/chatbot-ui)

## License
This project is released under the [Apache 2.0 license](https://github.com/vectorch-ai/ScaleLLM/blob/main/LICENSE).