# ScaleLLM: An efficient LLM Inference solution

> **WARNING**: ScaleLLM is currently in the active development stage and may not yet provide the optimal level of inference efficiency. We are fully dedicated to continuously enhancing its efficiency while also adding more features.

In the coming weeks, we have exciting plans to focus on [**_speculative decoding_**](https://github.com/orgs/vectorch-ai/projects/1) and [**_stateful conversation_**](https://github.com/orgs/vectorch-ai/projects/2), alongside further kernel optimizations. We appreciate your understanding and look forward to delivering an even better solution.


## Latest News:
* [11/2023] - First official release with support for popular open-source models.


## Table of contents

- [Overview](#overview)
- [Get Started](#get-started)
  - [Docker Container](#docker-container)
  - [Docker Compose](#docker-compose)
- [Usage Examples](#usage-examples)
- [Supported Models](#supported-models)
- [Quatization](#quatization)
- [Limitations](#limitations)
- [Contributing](#Contributing)


## Overview

ScaleLLM is a high-performance inference system for large language models, designed for production environments. It supports most popular open-source models, including Llama2, Bloom, GPT-NeoX, and more.

## Key Features

- **High Performance**: ScaleLLM is optimized for high-performance LLM inference.
- **Tensor Parallelism**: Utilizes tensor parallelism for efficient model execution.
- **OpenAI-compatible API** Efficient golang rest api server that compatible with OpenAI.
- **Huggingface models Integration** Seamless integration with most popular HF models.
- **Customizable**: Offers flexibility for customization to meet your specific needs.
- **Production Ready**: Designed to be deployed in production environments.

## Getting Started

The easiest way to get started with our project is by using the official Docker images. If you don't have Docker installed, please follow the installation instructions for your platform.

### Docker Installation

You can download and install Docker from the official website: [Docker Installation](https://docs.docker.com/get-docker/).

### Docker Container

Once you have Docker installed, you can run our project's Docker container using the following command:

```bash
docker run -it --gpus=all --net=host \
  -v $HOME/.cache/huggingface/hub:/models \
  -e HF_MODEL_ID=TheBloke/Llama-2-7B-chat-AWQ \
  -e DEVICE=cuda:0 \
  docker.io/vectorchai/scalellm:latest --logtostderr
```
Note: To use GPUs, you need to install the NVIDIA Container Toolkit. 

This command starts the Docker container with GPU support and various configuration options.

- `HF_MODEL_ID` specifies which Hugging Face model you want to run.
- `HF_MODEL_REVISION` specifies which Hugging Face model revision you want to run. by default, it is set to `"main"`.
- `HF_MODEL_ALLOW_PATTERN` specifies which types of files are allowed to be downloaded. by default, it is set to `"*.json,*.safetensors,*.model"`.
- `DEVICE` specifies the device on which this model should run. by default, it is set to `"auto"`.
- `HUGGING_FACE_HUB_TOKEN` specifies the token from [huggingface](https://huggingface.co/settings/tokens) for gated models.

### Ports and Endpoints

After running the Docker container, two ports are exposed:

1. **Port 8888 for gRPC Server**:

   The gRPC server is served on 0.0.0.0:8888 by default. You can use gRPC to interact with the service.

2. **Port 9999 for HTTP Server**:

   The simple HTTP server for instrument will be served on 0.0.0.0:9999 by default. This server provides various endpoints for managing and monitoring the service:

   - Use `curl localhost:9999/health` to check the health status of the service.
   - Use `curl localhost:9999/metrics` to export metrics.
   - Use `curl localhost:9999/gflags` to list all available gflags for configuration.

### Rest API Server

You can also start a REST API gateway using the following command:

```bash
docker run -it --net=host \
  docker.io/vectorchai/scalellm-gateway:latest --logtostderr
```

The exposed port is 8080. You can use REST API requests to interact with the system.

### Local Chatbot UI

A local chat Chatbot UI is also available. You can start it with the following command:

```bash
docker run -it --net=host \
  -e OPENAI_API_HOST=http://127.0.0.1:8080 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  docker.io/vectorchai/chatbot-ui:latest
```

## Docker Compose

You can run all the above services together using Docker Compose. Use the following command:

```bash
curl https://raw.githubusercontent.com/vectorch-ai/ScaleLLM/main/scalellm.yml -sSf > scalellm_compose.yml
HF_MODEL_ID=TheBloke/Llama-2-7B-chat-AWQ DEVICE=cuda docker compose -f ./scalellm_compose.yml up
```

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

## Supported Models

|   Models   | Tensor Parallel | Quantization | HF models examples |
| :--------: | :-------------: | :----------: | :---------------------------:|
|   Llama2   |       Yes       |     Yes      | [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), [TheBloke/Llama-2-13B-chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ), [TheBloke/Llama-2-70B-AWQ](https://huggingface.co/TheBloke/Llama-2-70B-AWQ) |
|   Aquila   |       Yes       |     Yes      | [BAAI/Aquila-7B](https://huggingface.co/BAAI/Aquila-7B), [BAAI/AquilaChat-7B](https://huggingface.co/BAAI/AquilaChat-7B) |
|   Bloom    |       Yes       |     Yes      | [bigscience/bloom](https://huggingface.co/bigscience/bloom) |
|   GPT_j    |       Yes       |     Yes      | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) |
|  GPT_NeoX  |       Yes       |     --      | [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) |
|    GPT2    |       Yes       |     --       | [gpt2](https://huggingface.co/gpt2)|
| InternLM   |       Yes       |     Yes      | [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) |
|  Mistral   |       Yes       |     Yes      | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
|    MPT     |       Yes       |     Yes      | [mosaicml/mpt-30b](https://huggingface.co/mosaicml/mpt-30b) |

If your model is not included in the supported list, we are more than willing to assist you. Please feel free to create a request for adding a new model on [GitHub Issues](https://github.com/vectorch-ai/ScaleLLM/issues).

## Quatization
Quantization is a crucial process for reducing the memory footprint of models. ScaleLLM offers support for two quantization techniques: Accurate Post-Training Quantization (**APTQ**) and Activation-aware Weight Quantization (**AWQ**), with seamless integration into the following libraries: autogptq, exllama, exllamav2, and awq. 

>By default, **exllamav2** is employed for GPTQ 4-bit quantization. However, you have the flexibility to choose a specific implementation by configuring the **"--qlinear_gptq_impl"** option, which allows you to select from exllama, exllamav2, or the automatic option.

## Limitations

There are several known limitations we are looking to address in the coming months, including:

- Only supports fast tokenizers with Hugging Face models.

## Contributing

If you have any questions or want to contribute, please don't hesitate to ask in our ["Discussions" forum](https://github.com/vectorch-ai/ScaleLLM/discussions) or join our ["Discord" chat room](https://discord.gg/PKe5gvBZfn). We welcome your input and contributions to make ScaleLLM even better.