from typing import Optional

from pydantic import BaseModel

from scalellm import Priority, Usage
from scalellm.serve.api_protocol import UsageInfo


def jsonify_model(obj: BaseModel):
    return obj.model_dump_json(exclude_unset=True)


def to_priority(priority: str) -> Priority:
    if priority == "low":
        return Priority.LOW
    if priority == "normal":
        return Priority.NORMAL
    if priority == "high":
        return Priority.HIGH
    return Priority.DEFAULT


def get_printable_token(logprob) -> str:
    return (
        logprob.token
        if logprob.finished_token
        else "".join(
            f"\\x{byte:02x}" for byte in logprob.token.encode("utf-8", errors="replace")
        )
    )



def to_api_usage(usage: Optional[Usage]) -> Optional[UsageInfo]:
    if usage is None:
        return None
    return UsageInfo(
        prompt_tokens=usage.num_prompt_tokens,
        total_tokens=usage.num_total_tokens,
        completion_tokens=usage.num_generated_tokens,
    )
