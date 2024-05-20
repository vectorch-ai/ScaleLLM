__version__ = "0.1.1"
try:
    # torch needs to be imported first, otherwise it will segfault upon import.
    import torch  # noqa: F401
except ImportError:
    pass

from scalellm._C import (LLMHandler, Message, Priority, RequestOutput,
                         SamplingParams, SequenceOutput, Status, StatusCode,
                         Usage, get_metrics)
from scalellm.llm import LLM
from scalellm.llm_engine import (AsyncLLMEngine, OutputAsyncStream,
                                 OutputStream, ValidationError)

__all__ = [
    "Message",
    "LLM",
    "AsyncLLMEngine",
    "OutputAsyncStream",
    "OutputStream",
    "ValidationError",
    "Priority",
    "RequestOutput",
    "SamplingParams",
    "SequenceOutput",
    "Status",
    "StatusCode",
    "Usage",
    "LLMHandler",
    "get_metrics",
]
