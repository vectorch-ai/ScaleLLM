import os
from typing import List, Optional, Union

from scalellm._C import (LLMHandler, Message, Priority, RequestOutput,
                         SamplingParams)
from scalellm.downloader import download_hf_model
from scalellm.errors import ValidationError


class LLM:
    def __init__(
        self,
        model: str,
        revision: Optional[str] = None,
        draft_model: Optional[str] = None,
        draft_revision: Optional[str] = None,
        allow_patterns: Optional[str] = None,
        cache_dir: Optional[str] = None,
        convert_to_safetensors: bool = False,
        devices: Optional[str] = None,
        draft_devices: Optional[str] = None,
        block_size: int = 16,
        max_cache_size: int = 0, # 0 means that cache size is caculated by available memory
        max_memory_utilization: float = 0.9,
        enable_prefix_cache: bool = True,
        enable_cuda_graph: bool = True,
        cuda_graph_max_seq_len: int = 2048,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        draft_cuda_graph_batch_sizes: Optional[List[int]] = None,
        max_tokens_per_batch: int = 409600,  # a big number to disable chunked prefill
        max_seqs_per_batch: int = 2048,  # a big number for better throughput
        num_speculative_tokens: int = 0,
        num_handling_threads: int = 4,
    ) -> None:
        # download hf model if it does not exist
        self._model = model
        self._draft_model = draft_model
        model_path = model
        if not os.path.exists(model_path):
            model_path = download_hf_model(
                repo_id=model_path,
                revision=revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                convert_to_safetensors=convert_to_safetensors,
            )
        draft_model_path = draft_model
        if draft_model_path is not None and not os.path.exists(draft_model_path):
            draft_model_path = download_hf_model(
                repo_id=draft_model_path,
                revision=draft_revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                convert_to_safetensors=convert_to_safetensors,
            )

        options = LLMHandler.Options()
        options.model_path = model_path
        options.devices = devices
        options.draft_model_path = draft_model_path
        options.draft_devices = draft_devices
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.enable_cuda_graph = enable_cuda_graph
        options.cuda_graph_max_seq_len = cuda_graph_max_seq_len
        options.cuda_graph_batch_sizes = cuda_graph_batch_sizes
        options.draft_cuda_graph_batch_sizes = draft_cuda_graph_batch_sizes
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.num_speculative_tokens = num_speculative_tokens
        options.num_handling_threads = num_handling_threads
        # create the LLM handler
        self._handler = LLMHandler(options)

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        priority: Priority = Priority.NORMAL,
        wait_for_schedule: bool = True,
    ) -> List[RequestOutput]:
        # use default sampling parameters if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()
        # convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params]

        if len(sampling_params) != len(prompts) and len(sampling_params) != 1:
            raise ValueError("The number of prompts and sampling parameters must match")

        outputs = [None] * len(prompts)

        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        # schedule the batch requests
        future = self._handler.schedule_batch_async(
            prompts, sampling_params, priority, False, callback
        )

        # wait for batch request to be scheduled
        if wait_for_schedule:
            future.wait()

        # run until all scheduled requsts complete
        self._handler.run_until_complete()

        # throw an exception if there is any error
        for index, output in enumerate(outputs):
            if output is None:
                raise RuntimeError("Request failed, no output received")
            if output.status is not None and not output.status.ok:
                raise ValidationError(output.status.code, output.status.message)
            # carry over the prompt to the output
            output.prompt = prompts[index]
        return outputs

    def apply_chat_template(self, messages: List[Message]) -> Optional[str]:
        return self._handler.apply_chat_template(messages)

    def encode(self, text: str) -> List[int]:
        return self._handler.encode(text)

    def decode(
        self, tokens: List[int], skip_special_tokens: bool = True
    ) -> Optional[str]:
        return self._handler.decode(tokens, skip_special_tokens)

    def __del__(self):
        if hasattr(self, "_handler"):
            self._handler.reset()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.__del__()
        return False
    
    def __repr__(self) -> str:
        if self._draft_model:
            return f"LLM(model={self._model}, draft_model={self._draft_model})"
        return f"LLM(model={self._model})"
