import time
from typing import List, Optional

import shortuuid

from scalellm import AsyncLLMEngine, LogProb, SamplingParams
from scalellm.serve.api_protocol import (CompletionLogProbs, CompletionRequest,
                                         CompletionResponse,
                                         CompletionResponseChoice,
                                         CompletionResponseStreamChoice,
                                         CompletionStreamResponse)
from scalellm.serve.common import (get_printable_token, jsonify_model,
                                   to_api_usage, to_priority)
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


def to_api_logprobs(
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
    choices = []
    prompt_len = len(request.prompt)
    for seq_output in output.outputs:
        choices.append(
            CompletionResponseChoice(
                index=seq_output.index,
                text=seq_output.text,
                logprobs=to_api_logprobs(seq_output.logprobs, prompt_len),
                finish_reason=seq_output.finish_reason,
            )
        )
    return CompletionResponse(
        id=request_id,
        object="text_completion",
        created=created_time,
        model=model,
        choices=choices,
        usage=to_api_usage(output.usage),
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

    include_usage = request.stream_options and request.stream_options.include_usage

    async def generate_stream_content():
        prompt_len = len(request.prompt)
        offsets = {}
        usage = None
        async for output in output_stream:
            for seq_output in output.outputs:
                cur_offset = offsets.setdefault(seq_output.index, prompt_len)
                offsets[seq_output.index] += len(seq_output.text)
                # send chunk with delta message
                if seq_output.text:
                    response = CompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=seq_output.index,
                                text=seq_output.text,
                                logprobs=to_api_logprobs(
                                    seq_output.logprobs, cur_offset
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    if include_usage:
                        response.usage = None
                    yield f"data: {jsonify_model(response)}\n\n"

                # send seperate chunk with finish reason
                if seq_output.finish_reason:
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
                    if include_usage:
                        response.usage = None
                    yield f"data: {jsonify_model(response)}\n\n"
            # record last usage info
            if output.usage:
                usage = output.usage

        # send additional chunk for usage info
        if include_usage and usage:
            response = CompletionStreamResponse(
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
