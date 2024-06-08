import time
from typing import List

import shortuuid

from scalellm import AsyncLLMEngine, Message, SamplingParams
from scalellm.serve.api_protocol import (ChatCompletionMessage,
                                         ChatCompletionRequest,
                                         ChatCompletionResponse,
                                         ChatCompletionResponseChoice,
                                         ChatCompletionResponseStreamChoice,
                                         ChatCompletionStreamResponse,
                                         ChatMessage, DeltaMessage, UsageInfo)
from scalellm.serve.common import (jsonify_model, to_api_chat_logprobs,
                                   to_priority)
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
    usage = None
    if output.usage:
        usage = UsageInfo(
            prompt_tokens=output.usage.num_prompt_tokens,
            total_tokens=output.usage.num_total_tokens,
            completion_tokens=output.usage.num_generated_tokens,
        )
    choices = []
    for seq_output in output.outputs:
        choices.append(
            ChatCompletionResponseChoice(
                index=seq_output.index,
                message=ChatMessage(role="assistant", content=seq_output.text),
                logprobs=to_api_chat_logprobs(seq_output.logprobs),
                finish_reason=seq_output.finish_reason,
            )
        )
    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=created_time,
        model=model,
        choices=choices,
        usage=usage,
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

    async def generate_stream_content():
        # to keep track of the first message sent
        first_message_sent = set()
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
                                logprobs=to_api_chat_logprobs(seq_output.logprobs),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {jsonify_model(response)}\n\n"
                    first_message_sent.add(index)
                # send chunk with delta message
                response = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    model=model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=seq_output.text),
                            logprobs=to_api_chat_logprobs(seq_output.logprobs),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {jsonify_model(response)}\n\n"
                # send the final chunk with finish reason
                if seq_output.finish_reason is not None:
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(),
                                logprobs=to_api_chat_logprobs(seq_output.logprobs),
                                finish_reason=seq_output.finish_reason,
                            )
                        ],
                    )
                    yield f"data: {jsonify_model(response)}\n\n"

            # send additional chunk for usage info
            if output.usage:
                response = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    model=model,
                    choices=[],
                    usage=UsageInfo(
                        prompt_tokens=output.usage.num_prompt_tokens,
                        total_tokens=output.usage.num_total_tokens,
                        completion_tokens=output.usage.num_generated_tokens,
                    ),
                )
                yield f"data: {jsonify_model(response)}\n\n"
        yield "data: [DONE]\n\n"

    return SafeStreamingResponse(
        content=generate_stream_content(), media_type="text/event-stream"
    )
