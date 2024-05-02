from typing import List

# Defined in scalellm/csrc/scalellm.cpp
class SamplingParameter:
    def __init__(self) -> None: ...
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: int

# Defined in scalellm/csrc/scalellm.cpp
class StoppingCriteria:
    def __init__(self) -> None: ...
    max_tokens: int
    eos_token_id: int
    ignore_eos_token: bool
    stop_token_ids: List[int]

# Defined in scalellm/csrc/llm.h
class LLM:
    def __init__(
        self,
        model_path: str,
        sampling_parameter: SamplingParameter,
        stopping_criteria: StoppingCriteria,
        max_seq_len: int,
        devices: str,
    ) -> None: ...
    def generate(self, batched_prompt: List[str]) -> None: ...
