import time
from typing import List, Optional

import shortuuid

from scalellm import (AsyncLLMEngine, LogProb, LogProbData, Message,
                      SamplingParams)
from scalellm.serve.api_protocol import (ChatCompletionLogProb,
                                         ChatCompletionLogProbData,
                                         ChatCompletionLogProbs,
                                         ChatCompletionMessage,
                                         ChatCompletionRequest,
                                         ChatCompletionResponse,
                                         ChatCompletionResponseChoice,
                                         ChatCompletionResponseStreamChoice,
                                         ChatCompletionStreamResponse,
                                         ChatMessage, DeltaMessage)
from scalellm.serve.common import (get_printable_token, jsonify_model,
                                   to_api_usage, to_priority)
from scalellm.serve.streaming_response import SafeStreamingResponse


def to_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
    sp = SamplingParams()
    sp.max_tokens = request.max_tokens
    sp.n = request.n
    # no echo for chat completion
    sp.echo = False
    sp.frequency_penalty = request.frequency_penalty
    sp.presence_penalty = request.presence_penalty
    sp.repetition_penalty = request.repetition_penalty
    sp.temperature = request.temperature
    sp.top_p = request.top_p
    sp.top_k = request.top_k
    sp.logprobs = request.logprobs
    sp.top_logprobs = request.top_logprobs
    sp.skip_special_tokens = request.skip_special_tokens
    sp.stop = request.stop
    sp.ignore_eos = request.ignore_eos
    sp.stop_token_ids = request.stop_token_ids
    return sp


def to_messages(messages: List[ChatCompletionMessage]) -> List[Message]:
    return [Message(role=msg.role, content=msg.content) for msg in messages]


def to_api_logprobdata(logprobdata: LogProbData) -> ChatCompletionLogProbData:
    return ChatCompletionLogProbData(
        token=get_printable_token(logprobdata),
        token_id=logprobdata.token_id,
        logprob=logprobdata.logprob,
        bytes=list(logprobdata.token.encode("utf-8", errors="replace")),
    )


def to_api_logprob(logprob: LogProb) -> ChatCompletionLogProb:
    top_logprobs = None
    if logprob.top_logprobs:
        top_logprobs = [to_api_logprobdata(d) for d in logprob.top_logprobs]
    return ChatCompletionLogProb(
        token=get_printable_token(logprob),
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


async def generate_chat_response(
    request: ChatCompletionRequest, engine: AsyncLLMEngine
) -> ChatCompletionResponse:
    assert not request.stream, "Non-streaming request expected"

    request_id = f"chatcmpl-{shortuuid.random()}"
    created_time = int(time.time())
    model = request.model

    sampling_params = to_sampling_params(request)
    priority = to_priority(request.priority)
    messages = to_messages(request.messages)

    output_stream = await engine.schedule_chat_async(
        messages, sampling_params, priority, request.stream
    )

    # only one output is expected for non-streaming request
    output = await output_stream.__anext__()
    choices = []
    for seq_output in output.outputs:
        choices.append(
            ChatCompletionResponseChoice(
                index=seq_output.index,
                message=ChatMessage(role="assistant", content=seq_output.text),
                logprobs=to_api_logprobs(seq_output.logprobs),
                finish_reason=seq_output.finish_reason,
            )
        )
    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=created_time,
        model=model,
        choices=choices,
        usage=to_api_usage(output.usage),
    )


async def generate_chat_stream_response(
    request: ChatCompletionRequest, engine: AsyncLLMEngine
) -> SafeStreamingResponse:
    assert request.stream, "Streaming request expected"

    request_id = f"chatcmpl-{shortuuid.random()}"
    created_time = int(time.time())
    model = request.model
    chunk_object_type = "chat.completion.chunk"

    sampling_params = to_sampling_params(request)
    priority = to_priority(request.priority)
    messages = to_messages(request.messages)

    output_stream = await engine.schedule_chat_async(
        messages, sampling_params, priority, request.stream
    )

    include_usage = request.stream_options and request.stream_options.include_usage

    async def generate_stream_content():
        # to keep track of the first message sent
        first_message_sent = set()
        usage = None
        async for output in output_stream:
            for seq_output in output.outputs:
                index = seq_output.index
                # send first chunk with role as assistant
                if index not in first_message_sent:
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(role="assistant", content=""),
                                logprobs=None,
                                finish_reason=None,
                            )
                        ],
                    )
                    if include_usage:
                        response.usage = None
                    yield f"data: {jsonify_model(response)}\n\n"
                    first_message_sent.add(index)
                # send chunk with delta message
                if seq_output.text:
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(content=seq_output.text),
                                logprobs=to_api_logprobs(seq_output.logprobs),
                                finish_reason=None,
                            )
                        ],
                    )
                    if include_usage:
                        response.usage = None
                    yield f"data: {jsonify_model(response)}\n\n"

                # send a seperate chunk with finish reason
                if seq_output.finish_reason:
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(),
                                logprobs=None,
                                finish_reason=seq_output.finish_reason,
                            )
                        ],
                    )
                    if include_usage:
                        response.usage = None
                    yield f"data: {jsonify_model(response)}\n\n"
            # record last usage info
            if output.usage:
                usage = output.usage

        # send additional chunk for usage info
        if include_usage and usage:
            response = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                model=model,
                choices=[],
                usage=to_api_usage(usage),
            )
            yield f"data: {jsonify_model(response)}\n\n"
        yield "data: [DONE]\n\n"

    return SafeStreamingResponse(
        content=generate_stream_content(), media_type="text/event-stream"
    )
