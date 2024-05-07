from enum import Enum
from typing import Callable, List, Optional

# Defined in scalellm/csrc/scalellm.cpp
def get_metrics() -> str: ...

class SamplingParams:
    def __init__(self) -> None: ...
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

class SequenceOutput:
    def __init__(self) -> None: ...
    index: int
    text: str
    finish_reason: Optional[str]

class RequestOutput:
    def __init__(self) -> None: ...
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

class ScheduleTask:
    def wait(self) -> None: ...
    def get(self) -> bool: ...

class LLMHandler:
    def __init__(self, model_path: str, devices: str) -> None: ...
    def schedule_async(
        self,
        prompt: str,
        sp: SamplingParams,
        priority: Priority,
        stream: bool,
        callback: Callable[[RequestOutput], bool],
    ) -> ScheduleTask: ...
    def schedule_chat_async(
        self,
        messages: List[Message],
        sp: SamplingParams,
        priority: Priority,
        stream: bool,
        callback: Callable[[RequestOutput], bool],
    ) -> ScheduleTask: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_until_complete(self) -> None: ...

# Defined in scalellm/csrc/llm.h
class LLM:
    def __init__(
        self,
        model_path: str,
        sampling_parameter: SamplingParams,
        max_seq_len: int,
        devices: str,
    ) -> None: ...
    def generate(self, batched_prompt: List[str]) -> None: ...
