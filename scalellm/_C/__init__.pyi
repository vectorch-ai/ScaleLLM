from enum import Enum
from typing import Callable, List, Optional

# Defined in scalellm/csrc/scalellm.cpp
def get_metrics() -> str: ...

class SamplingParams:
    def __init__(
        self,
        max_tokens: int = 16,
        n: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        logprobs: bool = False,
        top_logprobs: int = 0,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> None: ...
    # number of tokens to generate. truncted to model's max context length.
    max_tokens: int
    # number of sequences to generate for each prompt.
    n: int
    # whether to include the original prompt in the completion response.
    echo: bool
    # frequency penalty to reduce the likelihood of generating the same word multiple times. values between [0.0, 2.0].
    frequency_penalty: float
    # presence penalty to reduce the likelihood of generating words already in the prompt. values between [-2.0, 2.0].
    presence_penalty: float
    # repetition penalty to penalize new tokens based on their occurence in the text.
    repetition_penalty: float
    # temperature of the sampling, between [0, 2].
    temperature: float
    # top_p sampling cutoff, between [0.0, 1.0].
    top_p: float
    # top_k sampling cutoff. default = 0 to disable.
    top_k: int
    # Whether to return log probabilities of the output tokens or not.
    logprobs: bool
    # An integer between 0 and 20 specifying the number of most likely tokens to return at each token position.
    top_logprobs: int
    #  ############ stopping criterias. ############
    # whether to skip special tokens in the output text. default = true.
    skip_special_tokens: bool
    # whether to ignore eos token when checking stopping criterias. default = false.
    ignore_eos: bool
    # the list of strings to stop generating further tokens.
    stop: Optional[List[str]]
    # the list of token ids to stop generating further tokens.
    stop_token_ids: Optional[List[int]]

class Message:
    def __init__(self, role: str, content: str) -> None: ...
    role: str
    content: str

class Priority(Enum):
    DEFAULT: Priority = ...
    LOW: Priority = ...
    NORMAL: Priority = ...
    HIGH: Priority = ...

class Usage:
    def __init__(self) -> None: ...
    num_prompt_tokens: int
    num_generated_tokens: int
    num_total_tokens: int

class LogProbData:
    def __init__(self) -> None: ...
    token: str
    token_id: int
    logprob: float

class LogProb:
    def __init__(self) -> None: ...
    token: str
    token_id: int
    logprob: float
    top_logprobs: Optional[List[LogProbData]]

class SequenceOutput:
    def __init__(self) -> None: ...
    index: int
    text: str
    finish_reason: Optional[str]
    logprobs: Optional[List[LogProb]]

class RequestOutput:
    def __init__(self) -> None: ...
    prompt: Optional[str]
    status: Optional[Status]
    outputs: List[SequenceOutput]
    usage: Optional[Usage]
    finished: bool

class StatusCode(Enum):
    OK: StatusCode = ...
    CANCELLED: StatusCode = ...
    UNKNOWN: StatusCode = ...
    INVALID_ARGUMENT: StatusCode = ...
    DEADLINE_EXCEEDED: StatusCode = ...
    RESOURCE_EXHAUSTED: StatusCode = ...
    UNAUTHENTICATED: StatusCode = ...
    UNAVAILABLE: StatusCode = ...
    UNIMPLEMENTED: StatusCode = ...

class Status:
    def __init__(self, code: StatusCode, message: str) -> None: ...
    @property
    def code(self) -> StatusCode: ...
    @property
    def message(self) -> str: ...
    @property
    def ok(self) -> bool: ...

class LLMHandler:
    class Options:
        def __init__(self) -> None: ...
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
    def schedule_async(
        self,
        prompt: str,
        sp: SamplingParams,
        priority: Priority,
        stream: bool,
        callback: Callable[[RequestOutput], bool],
    ) -> None: ...
    def schedule_chat_async(
        self,
        messages: List[Message],
        sp: SamplingParams,
        priority: Priority,
        stream: bool,
        callback: Callable[[RequestOutput], bool],
    ) -> None: ...
    def schedule_batch_async(
        self,
        prompts: List[str],
        sps: List[SamplingParams],
        priority: Priority,
        stream: bool,
        callback: Callable[[int, RequestOutput], bool],
    ) -> None: ...
    def schedule_batch_chat_async(
        self,
        conversations: List[List[Message]],
        sps: List[SamplingParams],
        priority: Priority,
        stream: bool,
        callback: Callable[[int, RequestOutput], bool],
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_until_complete(self) -> None: ...
    def reset(self) -> None: ...
    # helper functions
    def apply_chat_template(self, messages: List[Message]) -> Optional[str]: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int], skip_special_tokens: bool) -> str: ...
