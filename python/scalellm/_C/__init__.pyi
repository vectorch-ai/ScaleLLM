from typing import Callable, List

# Defined in scalellm/csrc/scalellm.cpp
class SamplingParams:
    def __init__(self) -> None: ...
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: int

class Statistics:
    def __init__(self) -> None: ...
    num_prompt_tokens: int
    num_generated_tokens: int
    num_total_tokens: int

class SequenceOutput:
    def __init__(self) -> None: ...
    index: int
    text: str
    # finish_reason: str

class RequestOutput:
    def __init__(self) -> None: ...
    outputs: List[SequenceOutput]
    stats: Statistics
    finished: bool

class _LLMEngine:
    def __init__(self, model_path: str, devices: str) -> None: ...
    def schedule_async(
        self, prompt: str, sp: SamplingParams, callback: Callable[[RequestOutput], bool]
    ) -> bool: ...
    def run_forever(self) -> bool: ...
    def run_until_complete(self) -> bool: ...
    def stop(self) -> None: ...

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
