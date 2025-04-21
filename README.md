<div align="center">

ScaleLLM
=================
<h3> An efficient LLM Inference solution </h3>

[![Discord][discord-shield]][discord-url]
[![X][x-shield]][x-url]
<br>
[![Docs][docs-shield]][docs-url]
[![PyPI][pypi-shield]][pypi-url]
[![downloads][github-downloads-shield]][github-downloads-link]
[![License][license-shield]][license-url]

[discord-shield]: https://dcbadge.vercel.app/api/server/PKe5gvBZfn?compact=true&style=flat
[discord-url]: https://discord.gg/PKe5gvBZfn
[x-shield]: https://img.shields.io/twitter/url?label=%20%40VectorchAI&style=social&url=https://x.com/VectorchAI
[x-url]: https://x.com/VectorchAI

[docs-shield]: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
[docs-url]: https://docs.vectorch.com/
[pypi-shield]: https://badge.fury.io/py/scalellm.svg
[pypi-url]: https://pypi.org/project/scalellm/
[github-downloads-shield]: https://img.shields.io/github/downloads/vectorch-ai/ScaleLLM/total?style=flat
[github-downloads-link]: https://github.com/vectorch-ai/ScaleLLM/releases
[build-shield]: https://github.com/vectorch-ai/ScaleLLM/actions/workflows/build.yml/badge.svg?branch=main
[build-url]:https://github.com/vectorch-ai/ScaleLLM/actions/workflows/build.yml
[license-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[license-url]: https://opensource.org/licenses/Apache-2.0

---

<div align="left">

[ScaleLLM](#) is a cutting-edge inference system engineered for large language models (LLMs), designed to meet the demands of production environments. It extends its support to a wide range of popular open-source models, including [Llama3.1](https://github.com/meta-llama/llama3), [Gemma2](https://github.com/google-deepmind/gemma), Bloom, GPT-NeoX, and more.

ScaleLLM is currently undergoing active development. We are fully committed to consistently enhancing its efficiency while also incorporating additional features. Feel free to explore our [**_Roadmap_**](https://github.com/vectorch-ai/ScaleLLM/issues/84) for more details.

## News:
* [06/2024] - ScaleLLM is now available on [PyPI](https://pypi.org/project/scalellm/). You can install it using `pip install scalellm`.
* [03/2024] - [Advanced features](#advanced-features) support for ✅ [CUDA graph](#cuda-graph), ✅ [prefix cache](#prefix-cache), ✅ [chunked prefill](#chunked-prefill) and ✅ [speculative decoding](#speculative-decoding).
* [11/2023] - [First release](https://github.com/vectorch-ai/ScaleLLM/releases/tag/v0.0.1) with support for popular [open-source models](#supported-models).

## Key Features

- [High Efficiency](): Excels in high-performance LLM inference, leveraging state-of-the-art techniques and technologies like [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Paged Attention](https://github.com/vllm-project/vllm), [Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference), and more.
- [Tensor Parallelism](): Utilizes tensor parallelism for efficient model execution.
- [OpenAI-compatible API](): An OpenAI-compatible REST API server that supports both chat and completions.
- [Huggingface models](): Seamless integration with most popular [HF models](#supported-models), supporting safetensors.
- [Customizable](): Offers flexibility for customization to meet your specific needs, and provides an easy way to add new models.
- [Production Ready](): Engineered with production environments in mind, ScaleLLM is equipped with robust system monitoring and management features to ensure a seamless deployment experience.

## Table of contents

- [Get Started](#get-started)
  - [Installation](#installation)
  - [Chatbot UI](#chatbot-ui)
  - [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
  - [CUDA Graph](#cuda-graph)
  - [Prefix Cache](#prefix-cache)
  - [Chunked Prefill](#chunked-prefill)
  - [Speculative Decoding](#speculative-decoding)
  - [Quantization](#quantization)
- [Supported Models](#supported-models)
- [Limitations](#limitations)
- [Contributing](#Contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Getting Started

ScaleLLM is available as a Python Wheel package on PyPI. You can install it using pip:
```bash
# Install scalellm with CUDA 12.4 and Pytorch 2.6.0
pip install -U scalellm
```

If you want to install ScaleLLM with different version of CUDA and Pytorch, you can pip install it with provding index URL of the version. For example, to install ScaleLLM with CUDA 12.1 and Pytorch 2.4.1, you can use the following command:

```bash
pip install -U scalellm -i https://whl.vectorch.com/cu121/torch2.4.1/
```

### Build from source
If no wheel package is available for your configuration, you can build ScaleLLM from source code. You can clone the repository and install it locally using the following commands:
```bash
python setup.py bdist_wheel
pip install dist/scalellm-*.whl
```

### OpenAI-Compatible Server
You can start the OpenAI-compatible REST API server with the following command:
```bash
python3 -m scalellm.serve.api_server --model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Chatbot UI

A local Chatbot UI is also available on [localhost:3000](localhost:3000). You can start it with [latest image](https://hub.docker.com/r/vectorchai/chatbot-ui/tags) using the following command:

```bash
docker pull docker.io/vectorchai/chatbot-ui:latest
docker run -it --net=host \
  -e OPENAI_API_HOST=http://127.0.0.1:8080 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  docker.io/vectorchai/chatbot-ui:latest
```

### Usage Examples
You can use ScaleLLM for offline batch inference, or online distributed inference. Below are some examples to help you get started. More examples can be found in the [examples](https://github.com/vectorch-ai/ScaleLLM/tree/main/examples) folder.

#### Chat Completions

Start rest api server with the following command:
```bash
python3 -m scalellm.serve.api_server --model=meta-llama/Meta-Llama-3.1-8B-Instruct
```

You can query the chat completions with curl:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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

or with openai python client:

```python {linenos=true}
import openai

client = openai.Client(
    base_url="http://localhost:8080/v1",
    api_key="EMPTY",
)

# List available models
models = client.models.list()
print("==== Available models ====")
for model in models.data:
    print(model.id)

# choose the first model
model = models.data[0].id

stream = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    stream=True,
)

print(f"==== Model: {model} ====")
for chunk in stream:
    choice = chunk.choices[0]
    delta = choice.delta
    if delta.content:
        print(delta.content, end="")
print()
```

#### Completions
Start rest api server with the following command:
```bash
python3 -m scalellm.serve.api_server --model=meta-llama/Meta-Llama-3.1-8B
```

For regular completions, you can use this example:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B",
    "prompt": "hello",
    "max_tokens": 32,
    "temperature": 0.7,
    "stream": true
  }'
```

```python {linenos=true}
import openai

client = openai.Client(
    base_url="http://localhost:8080/v1",
    api_key="EMPTY",
)

# List available models
models = client.models.list()

print("==== Available models ====")
for model in models.data:
    print(model.id)

# choose the first model
model = models.data[0].id

stream = client.completions.create(
    model=model,
    prompt="hello",
    max_tokens=32,
    temperature=0.7,
    stream=True,
)

print(f"==== Model: {model} ====")
for chunk in stream:
    choice = chunk.choices[0]
    if choice.text:
        print(choice.text, end="")
print()
```

## Advanced Features
### CUDA Graph
CUDA Graph can improve performance by reducing the overhead of launching kernels. ScaleLLM supports CUDA Graph for decoding by default. In addition, It also allows user to specify which batch size to capture by setting the `--cuda_graph_batch_sizes` flag.

for example:
```bash
python3 -m scalellm.serve.api_server \
  --model=meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable_cuda_graph=true \
  --cuda_graph_batch_sizes=1,2,4,8
```

The limitations of CUDA Graph could cause problems during development and debugging. If you encounter any issues related to it, you can disable CUDA Graph by setting the `--enable_cuda_graph=false` flag.

### Prefix Cache
The KV cache is a technique that caches the intermediate kv states to avoid redundant computation during LLM inference. Prefix cache extends this idea by allowing kv caches with the same prefix to be shared among different requests.

ScaleLLM supports Prefix Cache and enables it by default. You can disable it by setting the `--enable_prefix_cache=false` flag.

### Chunked Prefill
Chunked Prefill splits a long user prompt into multiple chunks and populates the remaining slots with decodes. This technique can improve decoding throughput and enhance the user experience caused by long stalls. However it may slightly increase Time to First Token (TTFT). ScaleLLM supports Chunked Prefill, and its behavior can be controlled by setting the following flags:
- `--max_tokens_per_batch`: The maximum tokens for each batch, default is 512.
- `--max_seqs_per_batch`: The maximum sequences for each batch, default is 128.

### Speculative Decoding
Speculative Decoding is a common used technique to speed up LLM inference without
changing distribution. During inference, it employs an economical approximation to generate speculative tokens, subsequently validated by the target model. For now, ScaleLLM supports Speculative Decoding with a draft model to generate draft tokens, which can be enabled by configuring a draft model and setting the speculative steps.

for example:
```bash
python3 -m scalellm.serve.api_server \
  --model=google/gemma-7b-it \
  --draft_model=google/gemma-2b-it \
  --num_speculative_tokens=5 \
  --device=cuda:0 \
  --draft_device=cuda:0
```

### Quantization
Quantization is a crucial process for reducing the memory footprint of models. ScaleLLM offers support for two quantization techniques: Accurate Post-Training Quantization ([GPTQ](https://arxiv.org/abs/2210.17323)) and Activation-aware Weight Quantization ([AWQ](https://arxiv.org/abs/2306.00978)), with seamless integration into the following libraries: autogptq and awq.


## Supported Models

|   Models   | Tensor Parallel | Quantization | Chat API | HF models examples |
| :--------: | :-------------: | :----------: | :------: | :---------------------------:|
|   Aquila   |       Yes       |     Yes      |    Yes   | [BAAI/Aquila-7B](https://huggingface.co/BAAI/Aquila-7B), [BAAI/AquilaChat-7B](https://huggingface.co/BAAI/AquilaChat-7B) |
|   Bloom    |       Yes       |     Yes      |    No    | [bigscience/bloom](https://huggingface.co/bigscience/bloom) |
|   Baichuan |       Yes       |     Yes      |    Yes   | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |
| ChatGLM4/3 |       Yes       |     Yes      |    Yes   | [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) |
|   Gemma2   |       Yes       |     Yes      |    Yes   | [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) |
|   GPT_j    |       Yes       |     Yes      |    No    | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) |
|  GPT_NeoX  |       Yes       |     Yes      |    No    | [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) |
|    GPT2    |       Yes       |     Yes      |    No    | [gpt2](https://huggingface.co/gpt2)|
| InternLM   |       Yes       |     Yes      |    Yes   | [internlm/internlm-7b](https://huggingface.co/internlm/internlm-7b) |
|   Llama3/2 |       Yes       |     Yes      |    Yes   | [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) |
|  Mistral   |       Yes       |     Yes      |    Yes   | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
|    MPT     |       Yes       |     Yes      |    Yes   | [mosaicml/mpt-30b](https://huggingface.co/mosaicml/mpt-30b) |
|   Phi2     |       Yes       |     Yes      |    No   | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) |
|   Qwen2    |       Yes       |     Yes      |    Yes   | [Qwen/Qwen-72B-Chat](https://huggingface.co/Qwen/Qwen-72B-Chat) |
|    Yi      |       Yes       |     Yes      |    Yes    |[01-ai/Yi-6B](https://huggingface.co/01-ai/Yi-6B), [01-ai/Yi-34B-Chat-4bits](https://huggingface.co/01-ai/Yi-34B-Chat-4bits), [01-ai/Yi-6B-200K](https://huggingface.co/01-ai/Yi-6B-200K) |

If your model is not included in the supported list, we are more than willing to assist you. Please feel free to create a request for adding a new model on [GitHub Issues](https://github.com/vectorch-ai/ScaleLLM/issues).

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

## License
This project is released under the [Apache 2.0 license](https://github.com/vectorch-ai/ScaleLLM/blob/main/LICENSE).
