"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

Usage:
python3 -m scalellm.serve.api_server
"""

import argparse
import time
from typing import List

import fastapi
import shortuuid
import uvicorn
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from scalellm import (AsyncLLMEngine, Message, OutputAsyncStream, OutputError,
                      Priority, SamplingParams, get_metrics)
from scalellm.serve.api_protocol import (ChatCompletionMessage,
                                         ChatCompletionRequest,
                                         ChatCompletionResponse,
                                         ChatCompletionResponseChoice,
                                         ChatCompletionResponseStreamChoice,
                                         ChatCompletionStreamResponse,
                                         ChatMessage, CompletionRequest,
                                         CompletionResponse,
                                         CompletionResponseChoice,
                                         CompletionResponseStreamChoice,
                                         CompletionStreamResponse,
                                         DeltaMessage, ErrorResponse,
                                         ModelList, UsageInfo)

app = fastapi.FastAPI()
llm_engine: AsyncLLMEngine = None


def jsonify_model(obj: BaseModel):
    return obj.model_dump_json(exclude_none=True)


def request_to_sampling_params(request: CompletionRequest) -> SamplingParams:
    sp = SamplingParams()
    sp.max_tokens = request.max_tokens
    sp.n = request.n
    sp.echo = request.echo
    sp.frequency_penalty = request.frequency_penalty
    sp.presence_penalty = request.presence_penalty
    sp.repetition_penalty = request.repetition_penalty
    sp.temperature = request.temperature
    sp.top_p = request.top_p
    sp.top_k = request.top_k
    sp.skip_special_tokens = request.skip_special_tokens
    sp.stop = request.stop
    sp.ignore_eos = request.ignore_eos
    sp.stop_token_ids = request.stop_token_ids
    return sp


def chat_request_to_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
    sp = SamplingParams()
    sp.max_tokens = request.max_tokens
    sp.n = request.n
    sp.echo = False
    sp.frequency_penalty = request.frequency_penalty
    sp.presence_penalty = request.presence_penalty
    sp.repetition_penalty = request.repetition_penalty
    sp.temperature = request.temperature
    sp.top_p = request.top_p
    sp.top_k = request.top_k
    sp.skip_special_tokens = request.skip_special_tokens
    sp.stop = request.stop
    sp.ignore_eos = request.ignore_eos
    sp.stop_token_ids = request.stop_token_ids
    return sp


def to_priority(priority: str) -> Priority:
    if priority == "low":
        return Priority.LOW
    if priority == "normal":
        return Priority.NORMAL
    if priority == "high":
        return Priority.HIGH
    return Priority.DEFAULT


def to_messages(messages: List[ChatCompletionMessage]) -> List[Message]:
    return [Message(msg.role, msg.content) for msg in messages]


@app.get("/metrics")
async def show_metrics() -> Response:
    metrics = get_metrics()
    return Response(content=metrics, status_code=200)


@app.get("/health")
async def show_health() -> Response:
    return Response(content="OK\n", status_code=200)


@app.get("/v1/models")
async def show_available_models():
    model_cards = []
    return ModelList(data=model_cards)


async def generate_chat_response(
    output_stream: OutputAsyncStream, model: str
) -> ChatCompletionResponse:
    request_id = f"chatcmpl-{shortuuid.random()}"
    created_time = int(time.time())

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
                finish_reason=seq_output.finish_reason,
            )
        )
    return ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model,
        choices=choices,
        usage=usage,
    )


async def generate_chat_stream_response(
    output_stream: OutputAsyncStream, model: str
) -> StreamingResponse:
    request_id = f"chatcmpl-{shortuuid.random()}"
    created_time = int(time.time())

    async def generate_stream_content():
        try:
            # to keep track of the first message sent
            first_message_sent = set()
            async for output in output_stream:
                for seq_output in output.outputs:
                    index = seq_output.index
                    # send first chunk with role as assistant
                    if index not in first_message_sent:
                        response = ChatCompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model,
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index, delta=DeltaMessage(role="assistant")
                                )
                            ],
                        )
                        yield f"data: {jsonify_model(response)}\n\n"
                        first_message_sent.add(index)
                    # send chunk with delta message
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(content=seq_output.text),
                            )
                        ],
                    )
                    yield f"data: {jsonify_model(response)}\n\n"
                    # send the final chunk with finish reason
                    if seq_output.finish_reason is not None:
                        response = ChatCompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model,
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        finish_reason=seq_output.finish_reason
                                    ),
                                )
                            ],
                        )
                        yield f"data: {jsonify_model(response)}\n\n"

                # send additional chunk for usage info
                if output.usage:
                    response = ChatCompletionStreamResponse(
                        id=request_id,
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
        except OutputError as e:
            yield f"error: {jsonify_model(ErrorResponse(object='error', message=e.message, code=e.code))}\n\n"

    return StreamingResponse(
        content=generate_stream_content(), media_type="text/event-stream"
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    sampling_params = chat_request_to_sampling_params(request)
    priority = to_priority(request.priority)
    messages = to_messages(request.messages)

    ouput_stream = await llm_engine.schedule_chat_async(
        messages, sampling_params, priority, request.stream
    )

    try:
        if request.stream:
            return await generate_chat_stream_response(ouput_stream, request.model)
        return await generate_chat_response(ouput_stream, request.model)
    except OutputError as e:
        return ErrorResponse(object="error", message=e.message, code=e.code)


async def generate_completion_response(
    output_stream: OutputAsyncStream, model: str
) -> CompletionResponse:
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())

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
            CompletionResponseChoice(
                index=seq_output.index,
                text=seq_output.text,
                finish_reason=seq_output.finish_reason,
            )
        )
    return CompletionResponse(
        id=request_id,
        created=created_time,
        model=model,
        choices=choices,
        usage=usage,
    )


async def generate_completion_stream_response(
    output_stream: OutputAsyncStream, model: str
) -> StreamingResponse:
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())

    async def generate_stream_content():
        try:
            async for output in output_stream:
                for seq_output in output.outputs:
                    # send chunk with delta message
                    response = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=seq_output.index,
                                text=seq_output.text,
                            )
                        ],
                    )
                    yield f"data: {jsonify_model(response)}\n\n"
                    # send the final chunk with finish reason
                    if seq_output.finish_reason is not None:
                        response = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model,
                            choices=[
                                CompletionResponseStreamChoice(
                                    index=seq_output.index,
                                    text="",
                                    finish_reason=seq_output.finish_reason,
                                )
                            ],
                        )
                        yield f"data: {jsonify_model(response)}\n\n"
            # send additional chunk for usage info
            if output.usage:
                response = CompletionStreamResponse(
                    id=request_id,
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
        except OutputError as e:
            yield f"error: {jsonify_model(ErrorResponse(object='error', message=e.message, code=e.code))}\n\n"

    return StreamingResponse(
        content=generate_stream_content(), media_type="text/event-stream"
    )


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion for the prompt"""
    sampling_params = request_to_sampling_params(request)
    priority = to_priority(request.priority)
    ouput_stream = await llm_engine.schedule_async(
        request.prompt, sampling_params, priority, request.stream
    )

    try:
        if request.stream:
            return await generate_completion_stream_response(
                ouput_stream, request.model
            )
        return await generate_completion_response(ouput_stream, request.model)
    except OutputError as e:
        return ErrorResponse(object="error", message=e.message, code=e.code)


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8080, help="port number")

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Name or path of the huggingface model.",
    )

    parser.add_argument(
        "--revision", type=str, default=None, help="Revision of the model."
    )

    parser.add_argument("--devices", type=str, default="auto", help="devices to use.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # initialize the LLM engine
    llm_engine = AsyncLLMEngine(args.model, args.devices)

    try:
        llm_engine.start()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        # stop the LLM engine
        llm_engine.stop()
