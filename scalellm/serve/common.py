from typing import List, Optional

from pydantic import BaseModel

from scalellm import LogProb, LogProbData, Priority
from scalellm.serve.api_protocol import (ChatCompletionLogProb,
                                         ChatCompletionLogProbData,
                                         ChatCompletionLogProbs,
                                         CompletionLogProbs)


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


def to_api_chat_logprobdata(logprobdata: LogProbData) -> ChatCompletionLogProbData:
    return ChatCompletionLogProbData(
        token=get_printable_token(logprobdata),
        token_id=logprobdata.token_id,
        logprob=logprobdata.logprob,
        bytes=list(logprobdata.token.encode("utf-8", errors="replace")),
    )


def to_api_chat_logprob(logprob: LogProb) -> ChatCompletionLogProb:
    top_logprobs = None
    if logprob.top_logprobs:
        top_logprobs = [to_api_chat_logprobdata(d) for d in logprob.top_logprobs]
    return ChatCompletionLogProb(
        token=get_printable_token(logprob),
        token_id=logprob.token_id,
        logprob=logprob.logprob,
        bytes=list(logprob.token.encode("utf-8", errors="replace")),
        top_logprobs=top_logprobs,
    )


def to_api_chat_logprobs(
    logprobs: Optional[List[LogProb]],
) -> Optional[ChatCompletionLogProbs]:
    if logprobs is None:
        return None
    return ChatCompletionLogProbs(
        content=[to_api_chat_logprob(logprob) for logprob in logprobs]
    )


def to_api_completion_logprobs(
    logprobs: Optional[List[LogProb]],
    offset: int,
) -> Optional[CompletionLogProbs]:
    if logprobs is None:
        return None

    text_offset, tokens, token_ids, token_logprobs, top_logprobs = [], [], [], [], []

    for logprob in logprobs:
        text_offset.append(offset)
        offset += len(logprob.token)
        tokens.append(get_printable_token(logprob))
        token_ids.append(logprob.token_id)
        token_logprobs.append(logprob.logprob)

        if logprob.top_logprobs:
            top_logprobs.append(
                {get_printable_token(top): top.logprob for top in logprob.top_logprobs}
            )
        else:
            top_logprobs.append(None)

    return CompletionLogProbs(
        text_offset=text_offset,
        tokens=tokens,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
    )
