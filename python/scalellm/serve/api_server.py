"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

Usage:
python3 -m scalellm.serve.api_server
"""

import argparse

import fastapi
import uvicorn
from fastapi.responses import JSONResponse, Response
from scalellm.serve.api_protocol import (ChatCompletionRequest,
                                         ChatCompletionResponse,
                                         CompletionRequest, CompletionResponse,
                                         ModelList, UsageInfo)

app = fastapi.FastAPI()


@app.get("/metrics")
async def show_metrics() -> JSONResponse:
    metrics = {}
    return JSONResponse(content=metrics)


@app.get("/health")
async def show_health() -> Response:
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    model_cards = []
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    choices = []
    usage = UsageInfo()
    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion for the prompt"""
    choices = []
    usage = UsageInfo()
    return CompletionResponse(
        model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8080, help="port number")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
