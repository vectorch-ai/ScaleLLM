import os
from typing import List, Optional, Union

from scalellm._C import LLMHandler, Priority, RequestOutput, SamplingParams
from scalellm.downloader import download_hf_model


class LLM:
    def __init__(
        self,
        model: str,
        revision: Optional[str] = None,
        draft_model: Optional[str] = None,
        draft_revision: Optional[str] = None,
        allow_patterns: Optional[str] = None,
        cache_dir: Optional[str] = None,
        devices: Optional[str] = None,
        draft_devices: Optional[str] = None,
        block_size: int = 16,
        max_cache_size: int = 10 * 1024 * 1024 * 1024,
        max_memory_utilization: float = 0.9,
        enable_prefix_cache: bool = True,
        enable_cuda_graph: bool = True,
        cuda_graph_max_seq_len: int = 2048,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        draft_cuda_graph_batch_sizes: Optional[List[int]] = None,
        max_tokens_per_batch: int = 512,
        max_seqs_per_batch: int = 128,
        num_speculative_tokens: int = 0,
        num_handling_threads:int = 4,
    ) -> None:
        # download hf model if it does not exist
        model_path = model
        if not os.path.exists(model_path):
            model_path = download_hf_model(
                repo_id=model_path,
                revision=revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
            )
        draft_model_path = draft_model
        if draft_model_path is not None and not os.path.exists(draft_model_path):
            draft_model_path = download_hf_model(
                repo_id=draft_model_path,
                revision=draft_revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
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
    ) -> List[RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            # use default sampling parameters
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        if len(sampling_params) != len(prompts):
            raise ValueError("The number of prompts and sampling parameters must match")

        outputs = [None] * len(prompts)
        for i in range(len(prompts)):

            def callback(output: RequestOutput, idx: int = i) -> bool:
                outputs[idx] = output
                return True

            self._handler.schedule_async(
                prompts[i], sampling_params[i], priority, False, callback
            )

        # run until all scheduled requsts complete
        self._handler.run_until_complete()
        return outputs
