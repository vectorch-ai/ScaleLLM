from enum import Enum
from typing import Callable, List, Optional

from scalellm._C.output import RequestOutput
from scalellm._C.sampling_params import SamplingParams

# Defined in csrc/llm_handler.cpp
class Message:
    def __init__(self, role: str, content: str) -> None: ...
    def __repr__(self) -> str: ...
    role: str
    content: str

class Priority(Enum):
    DEFAULT: Priority = ...
    LOW: Priority = ...
    NORMAL: Priority = ...
    HIGH: Priority = ...

class Future:
    def wait(self) -> None: ...
    def get(self) -> bool: ...

class BatchFuture:
    def wait(self) -> None: ...
    def get(self) -> List[bool]: ...

class LLMHandler:
    class Options:
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        model_path: str
        devices: Optional[str]
        draft_model_path: Optional[str]
        draft_devices: Optional[str]
        block_size: int
        max_cache_size: int
        max_memory_utilization: float
        enable_prefix_cache: bool
        enable_cuda_graph: bool
        cuda_graph_max_seq_len: int
        cuda_graph_batch_sizes: Optional[List[int]]
        draft_cuda_graph_batch_sizes: Optional[List[int]]
        max_tokens_per_batch: int
        max_seqs_per_batch: int
        num_speculative_tokens: int
        num_handling_threads: int

    def __init__(self, options: Options) -> None: ...
    def __repr__(self) -> str: ...
    def schedule_async(
        self,
        prompt: str,
        sp: SamplingParams,
        priority: Priority,
        stream: bool,
        callback: Callable[[RequestOutput], bool],
    ) -> Future: ...
    def schedule_chat_async(
        self,
        messages: List[Message],
        sp: SamplingParams,
        priority: Priority,
        stream: bool,
        callback: Callable[[RequestOutput], bool],
    ) -> Future: ...
    def schedule_batch_async(
        self,
        prompts: List[str],
        sps: List[SamplingParams],
        priority: Priority,
        stream: bool,
        callback: Callable[[int, RequestOutput], bool],
    ) -> BatchFuture: ...
    def schedule_batch_chat_async(
        self,
        conversations: List[List[Message]],
        sps: List[SamplingParams],
        priority: Priority,
        stream: bool,
        callback: Callable[[int, RequestOutput], bool],
    ) -> BatchFuture: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_until_complete(self) -> None: ...
    def reset(self) -> None: ...
    # helper functions
    def apply_chat_template(self, messages: List[Message]) -> Optional[str]: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int], skip_special_tokens: bool) -> str: ...
