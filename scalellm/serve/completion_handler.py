import time

import shortuuid

from scalellm import AsyncLLMEngine, SamplingParams
from scalellm.serve.api_protocol import (CompletionRequest, CompletionResponse,
                                         CompletionResponseChoice,
                                         CompletionResponseStreamChoice,
                                         CompletionStreamResponse, UsageInfo)
from scalellm.serve.common import (jsonify_model, to_api_completion_logprobs,
                                   to_priority)
from scalellm.serve.streaming_response import SafeStreamingResponse


def to_sampling_params(request: CompletionRequest) -> SamplingParams:
    sp = SamplingParams()
    sp.max_tokens = request.max_tokens
    sp.n = request.n
    sp.best_of = request.best_of
    sp.echo = request.echo
    sp.frequency_penalty = request.frequency_penalty
    sp.presence_penalty = request.presence_penalty
    sp.repetition_penalty = request.repetition_penalty
    sp.temperature = request.temperature
    sp.top_p = request.top_p
    sp.top_k = request.top_k
    if request.logprobs:
        sp.logprobs = True
        sp.top_logprobs = request.logprobs
    sp.skip_special_tokens = request.skip_special_tokens
    sp.stop = request.stop
    sp.ignore_eos = request.ignore_eos
    sp.stop_token_ids = request.stop_token_ids
    return sp


async def generate_completion_response(
    request: CompletionRequest, engine: AsyncLLMEngine
) -> CompletionResponse:
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())
    model = request.model

    sampling_params = to_sampling_params(request)
    priority = to_priority(request.priority)
    output_stream = await engine.schedule_async(
        request.prompt,
        sampling_params=sampling_params,
        priority=priority,
        stream=False,
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
    prompt_len = len(request.prompt)
    for seq_output in output.outputs:
        choices.append(
            CompletionResponseChoice(
                index=seq_output.index,
                text=seq_output.text,
                logprobs=to_api_completion_logprobs(seq_output.logprobs, prompt_len),
                finish_reason=seq_output.finish_reason,
            )
        )
    return CompletionResponse(
        id=request_id,
        object="text_completion",
        created=created_time,
        model=model,
        choices=choices,
        usage=usage,
    )


async def generate_completion_stream_response(
    request: CompletionRequest, engine: AsyncLLMEngine
) -> SafeStreamingResponse:
    assert request.stream, "non-streaming request is not supported"

    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())
    chunk_object_type = "text_completion"
    model = request.model

    sampling_params = to_sampling_params(request)
    priority = to_priority(request.priority)
    output_stream = await engine.schedule_async(
        request.prompt,
        sampling_params=sampling_params,
        priority=priority,
        stream=request.stream,
    )

    async def generate_stream_content():
        prompt_len = len(request.prompt)
        offsets = {}
        async for output in output_stream:
            for seq_output in output.outputs:
                cur_offset = offsets.setdefault(seq_output.index, prompt_len)
                offsets[seq_output.index] += len(seq_output.text)
                # send chunk with delta message
                response = CompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    model=model,
                    choices=[
                        CompletionResponseStreamChoice(
                            index=seq_output.index,
                            text=seq_output.text,
                            logprobs=to_api_completion_logprobs(
                                seq_output.logprobs, cur_offset
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {jsonify_model(response)}\n\n"
                # send the final chunk with finish reason
                if seq_output.finish_reason is not None:
                    response = CompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=seq_output.index,
                                text="",
                                logprobs=None,
                                finish_reason=seq_output.finish_reason,
                            )
                        ],
                    )
                    yield f"data: {jsonify_model(response)}\n\n"
        # send additional chunk for usage info
        if output.usage:
            response = CompletionStreamResponse(
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
