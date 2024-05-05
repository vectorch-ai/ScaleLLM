"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

Usage:
python3 -m scalellm.serve.api_server
"""

import argparse
import time

import fastapi
import shortuuid
import uvicorn
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from scalellm import LLMEngine, SamplingParams
from scalellm.serve.api_protocol import (ChatCompletionRequest,
                                         ChatCompletionResponse,
                                         CompletionRequest,
                                         CompletionResponseStreamChoice,
                                         CompletionStreamResponse,
                                         ErrorResponse, ModelList, UsageInfo)

app = fastapi.FastAPI()
llm_engine: LLMEngine = None


def jsonify_model(obj: BaseModel):
    return obj.model_dump_json(exclude_none=True)


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
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())

    sampling_params = SamplingParams()
    ouput_stream = await llm_engine.schedule_async(request.prompt, sampling_params)

    async def generate_stream_response():
        async for output in ouput_stream:
            usage = None
            if output.finished:
                usage = UsageInfo(
                    prompt_tokens=output.stats.num_prompt_tokens,
                    total_tokens=output.stats.num_total_tokens,
                    completion_tokens=output.stats.num_generated_tokens,
                )
            for seq_output in output.outputs:
                choice = CompletionResponseStreamChoice(
                    index=seq_output.index,
                    text=seq_output.text,
                    finish_reason=None,
                )

                response = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[choice],
                    usage=usage,
                )
                yield f"data: {jsonify_model(response)}\n\n"
        yield "data: [DONE]\n\n"

    try:
        return StreamingResponse(
            content=generate_stream_response(), media_type="text/event-stream"
        )
    except Exception as e:
        return ErrorResponse(object="error", message=str(e), code=500)


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
    llm_engine = LLMEngine(args.model, args.devices)
    llm_engine.run_forever()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        # stop the LLM engine
        llm_engine.stop()
