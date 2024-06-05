import json
from functools import partial
from http import HTTPStatus
from typing import AsyncIterable, Awaitable, Callable, Mapping, Optional

from anyio import create_task_group
from fastapi.responses import Response

from scalellm import ValidationError
from scalellm.serve.api_protocol import ErrorResponse


class SafeStreamingResponse(Response):
    body_iterator: AsyncIterable[str]

    def __init__(
        self,
        content: AsyncIterable[str],
        status_code: HTTPStatus = HTTPStatus.OK,
        error_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
    ) -> None:
        self.body_iterator = content
        self.status_code = status_code.value
        self.error_code = error_code.value
        self.media_type = self.media_type if media_type is None else media_type
        self.init_headers(headers)
        self.background = None

    async def listen_for_disconnect(self, receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break

    async def send_first(self, status_code, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": self.raw_headers,
            }
        )

    async def send_last(self, send) -> None:
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def send_chunk(self, chunk, send) -> None:
        await send(
            {
                "type": "http.response.body",
                "body": chunk,
                "more_body": True,
            }
        )

    async def stream_response(self, send) -> None:
        # defer the first send for right status code
        first_sent = False
        try:
            async for chunk in self.body_iterator:
                if not first_sent:
                    await self.send_first(self.status_code, send)
                    first_sent = True
                chunk = chunk.encode(self.charset)
                await self.send_chunk(chunk, send)
        except ValidationError as e:
            if not first_sent:
                await self.send_first(self.error_code, send)
                first_sent = True
            error_chunk = json.dumps(
                {"error": ErrorResponse(message=e.message, code=e.code).dict()},
                separators=(",", ":"),
            ).encode("utf-8")
            await self.send_chunk(error_chunk, send)
        finally:
            if not first_sent:
                await self.send_first(self.status_code, send)
            await self.send_last(send)

    async def __call__(self, scope, receive, send) -> None:
        async with create_task_group() as task_group:

            async def wrap(func: Callable[[], Awaitable[None]]) -> None:
                await func()
                task_group.cancel_scope.cancel()

            task_group.start_soon(wrap, partial(self.stream_response, send))
            await wrap(partial(self.listen_for_disconnect, receive))
