"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

Usage:
python3 -m scalellm.serve.api_server
"""


import fastapi
import uvicorn
from fastapi.responses import Response
from pydantic import BaseModel
from scalellm import AsyncLLMEngine, OutputError, get_metrics
from scalellm.serve.api_protocol import (ChatCompletionRequest,
                                         CompletionRequest, ErrorResponse,
                                         ModelList)
from scalellm.serve.chat_handler import (generate_chat_response,
                                         generate_chat_stream_response)
from scalellm.serve.completion_handler import (
    generate_completion_response, generate_completion_stream_response)
from scalellm.serve.server_args import parse_args

app = fastapi.FastAPI()
llm_engine: AsyncLLMEngine = None


def jsonify_model(obj: BaseModel):
    return obj.model_dump_json(exclude_unset=True)


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


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    try:
        if request.stream:
            return await generate_chat_stream_response(request, llm_engine)
        return await generate_chat_response(request, llm_engine)
    except OutputError as e:
        return ErrorResponse(object="error", message=e.message, code=e.code)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion for the prompt"""
    try:
        if request.stream:
            return await generate_completion_stream_response(request, llm_engine)
        return await generate_completion_response(request, llm_engine)
    except OutputError as e:
        return ErrorResponse(object="error", message=e.message, code=e.code)


def parse_batch_sizes(batch_sizes_str):
    if batch_sizes_str is None:
        return None
    return [int(size) for size in batch_sizes_str.split(",")]


if __name__ == "__main__":
    args = parse_args()
    # initialize the LLM engine
    llm_engine = AsyncLLMEngine(
        model=args.model,
        revision=args.revision,
        devices=args.devices,
        draft_model=args.draft_model,
        draft_revision=args.draft_revision,
        draft_devices=args.draft_devices,
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
    )

    try:
        llm_engine.start()
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:
        pass
    finally:
        # stop the LLM engine
        llm_engine.stop()
