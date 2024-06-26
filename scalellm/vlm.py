import os
from typing import List, Optional

import torch

from scalellm._C import Priority, RequestOutput, SamplingParams, VLMHandler
from scalellm.downloader import download_hf_model
from scalellm.errors import ValidationError


class VLM:
    def __init__(
        self,
        model: str,
        revision: Optional[str] = None,
        allow_patterns: Optional[str] = None,
        cache_dir: Optional[str] = None,
        convert_to_safetensors: bool = False,
        devices: Optional[str] = None,
        block_size: int = 16,
        max_cache_size: int = 20 * 1024 * 1024 * 1024,
        max_memory_utilization: float = 0.9,
        enable_prefix_cache: bool = True,
        enable_cuda_graph: bool = True,
        cuda_graph_max_seq_len: int = 2048,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        max_tokens_per_batch: int = 409600,  # a big number to disable chunked prefill
        max_seqs_per_batch: int = 2048,  # a big number for better throughput
        num_handling_threads: int = 4,
        # vision encoder configuration
        image_input_type: Optional[str] = None,
        image_token_id: Optional[int] = None,
        image_input_shape: Optional[str] = None,
        image_feature_size: Optional[int] = None,
    ) -> None:
        # download hf model if it does not exist
        self._model = model
        model_path = model
        if not os.path.exists(model_path):
            model_path = download_hf_model(
                repo_id=model_path,
                revision=revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                convert_to_safetensors=convert_to_safetensors,
            )

        options = VLMHandler.Options()
        options.model_path = model_path
        options.devices = devices
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.enable_cuda_graph = enable_cuda_graph
        options.cuda_graph_max_seq_len = cuda_graph_max_seq_len
        options.cuda_graph_batch_sizes = cuda_graph_batch_sizes
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.num_handling_threads = num_handling_threads
        options.image_input_type = image_input_type
        options.image_token_id = image_token_id
        options.image_input_shape = image_input_shape
        options.image_feature_size = image_feature_size
        # create the LLM handler
        self._handler = VLMHandler(options)

    def generate(
        self,
        image: torch.Tensor = None,
        prompt: str = None,
        sampling_params: Optional[SamplingParams] = None,
        priority: Priority = Priority.NORMAL,
        wait_for_schedule: bool = True,
    ) -> RequestOutput:
        # use default sampling parameters if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()

        output = None
        def callback(async_output: RequestOutput) -> bool:
            #output = async_output
            return True

        # schedule the batch requests
        future = self._handler.schedule_async(
            image, prompt, sampling_params, priority, False, callback
        )

        # wait for batch request to be scheduled
        if wait_for_schedule:
            future.wait()

        # run until all scheduled requsts complete
        self._handler.run_until_complete()
       
        # throw an exception if there is any error
        if output is None:
            raise RuntimeError("Request failed, no output received")
        if output.status is not None and not output.status.ok:
            raise ValidationError(output.status.code, output.status.message)
            # carry over the prompt to the output
        output.prompt = prompt
        return output

    def encode(self, text: str) -> List[int]:
        return self._handler.encode(text)

    def decode(
        self, tokens: List[int], skip_special_tokens: bool = True
    ) -> Optional[str]:
        return self._handler.decode(tokens, skip_special_tokens)

    def __del__(self):
        self._handler.reset()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.__del__()
        return False
    
    def __repr__(self) -> str:
        if self._draft_model:
            return f"VLM(model={self._model}, draft_model={self._draft_model})"
        return f"VLM(model={self._model})"
