from typing import List, Optional

from pydantic import BaseModel

from scalellm import LogProb, LogProbData, Priority
from scalellm.serve.api_protocol import (ChatCompletionLogProb,
                                         ChatCompletionLogProbData,
                                         ChatCompletionLogProbs)


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


def to_api_logprobdata(logprobdata: LogProbData) -> ChatCompletionLogProbData:
    return ChatCompletionLogProbData(
        token=logprobdata.token,
        token_id=logprobdata.token_id,
        logprob=logprobdata.logprob,
        bytes=list(logprobdata.token.encode("utf-8", errors="replace")),
    )


def to_api_logprob(logprob: LogProb) -> ChatCompletionLogProb:
    top_logprobs = None
    if logprob.top_logprobs:
        top_logprobs = [to_api_logprobdata(d) for d in logprob.top_logprobs]
    return ChatCompletionLogProb(
        token=logprob.token,
        token_id=logprob.token_id,
        logprob=logprob.logprob,
        bytes=list(logprob.token.encode("utf-8", errors="replace")),
        top_logprobs=top_logprobs,
    )


def to_api_logprobs(
    logprobs: Optional[List[LogProb]],
) -> Optional[ChatCompletionLogProbs]:
    if logprobs is None:
        return None
    return ChatCompletionLogProbs(
        content=[to_api_logprob(logprob) for logprob in logprobs]
    )
