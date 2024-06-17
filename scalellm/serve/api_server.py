"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

Usage:
python3 -m scalellm.serve.api_server
"""

import os
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import fastapi
import uvicorn
from fastapi.responses import JSONResponse, Response

from scalellm import AsyncLLMEngine, ValidationError, get_metrics
from scalellm.serve.api_protocol import (ChatCompletionRequest,
                                         CompletionRequest, ErrorResponse,
                                         ModelCard, ModelList, ModelPermission)
from scalellm.serve.chat_handler import (generate_chat_response,
                                         generate_chat_stream_response)
from scalellm.serve.completion_handler import (
    generate_completion_response, generate_completion_stream_response)
from scalellm.serve.server_args import parse_args

app = fastapi.FastAPI()
llm_engine: AsyncLLMEngine = None
models = None


def create_error_response(
    message: str, code: int, status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
) -> JSONResponse:
    return JSONResponse(
        {"error": ErrorResponse(message=message, code=code).dict()},
        status_code=status_code.value,
    )


def check_model(request) -> Optional[JSONResponse]:
    if request.model not in models:
        return create_error_response(
            message=f"The model `{request.model}` does not exist.",
            code=404,
            status_code=HTTPStatus.NOT_FOUND,
        )
    return None


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, e):
    return create_error_response(e.message, e.code)


@app.get("/metrics")
async def show_metrics() -> Response:
    metrics = get_metrics()
    return Response(content=metrics)


@app.get("/health")
async def show_health() -> Response:
    return Response(content="OK\n")


@app.get("/v1/models")
async def show_available_models():
    model_cards = [ModelCard(id=model_id, permission=[ModelPermission()])]
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_response = check_model(request)
    if error_response is not None:
        return error_response

    if request.stream:
        return await generate_chat_stream_response(request, llm_engine)
    return await generate_chat_response(request, llm_engine)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion for the prompt"""
    error_response = check_model(request)
    if error_response is not None:
        return error_response

    # results cannot be streamed when best_of != n
    stream = request.stream and (
        request.best_of is None or request.n == request.best_of
    )

    if stream:
        return await generate_completion_stream_response(request, llm_engine)
    return await generate_completion_response(request, llm_engine)


def parse_batch_sizes(batch_sizes_str):
    if batch_sizes_str is None:
        return None
    return [int(size) for size in batch_sizes_str.split(",")]


if __name__ == "__main__":
    args = parse_args()
    # set the model_id
    if args.model_id is not None:
        #  use the model_id provided by the user
        model_id = args.model_id
    elif os.path.exists(args.model):
        # use the directory name of the model path
        model_id = Path(args.model).stem
    else:
        # model is model name
        model_id = args.model
    models = [model_id]

    # initialize the LLM engine
    llm_engine = AsyncLLMEngine(
        model=args.model,
        revision=args.revision,
        devices=args.devices,
        draft_model=args.draft_model,
        draft_revision=args.draft_revision,
        draft_devices=args.draft_devices,
        cache_dir=args.cache_dir,
        convert_to_safetensors=args.convert_to_safetensors,
        block_size=args.block_size,
        max_cache_size=args.max_cache_size,
        max_memory_utilization=args.max_memory_utilization,
        enable_prefix_cache=args.enable_prefix_cache,
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_max_seq_len=args.cuda_graph_max_seq_len,
        cuda_graph_batch_sizes=parse_batch_sizes(args.cuda_graph_batch_sizes),
        draft_cuda_graph_batch_sizes=parse_batch_sizes(
            args.draft_cuda_graph_batch_sizes
        ),
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_seqs_per_batch=args.max_seqs_per_batch,
        num_speculative_tokens=args.num_speculative_tokens,
        num_handling_threads=args.num_handling_threads,
    )

    try:
        llm_engine.start()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # stop the LLM engine
        llm_engine.stop()
