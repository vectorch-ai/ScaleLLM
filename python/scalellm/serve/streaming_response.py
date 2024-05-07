from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from scalellm.llm_engine import OutputError


class SafeStreamingResponse(StreamingResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        try:
            async for chunk in self.body_iterator:
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
        except OutputError as e:
            await send(
                {
                    "type": "http.response.body",
                    "body": str(e).encode("utf-8"),
                    "more_body": True,
                }
            )

        await send({"type": "http.response.body", "body": b"", "more_body": False})
