from enum import Enum
from typing import List, Optional

# Defined in csrc/output.cpp
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
    finished_token: bool

class LogProb:
    def __init__(self) -> None: ...
    token: str
    token_id: int
    logprob: float
    finished_token: bool
    top_logprobs: Optional[List[LogProbData]]

class SequenceOutput:
    def __init__(self) -> None: ...
    index: int
    text: str
    token_ids: List[int]
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
