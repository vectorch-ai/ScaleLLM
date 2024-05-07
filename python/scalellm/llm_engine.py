import asyncio
import os
import queue
from typing import List

from scalellm._C import (
    LLMHandler,
    Message,
    Priority,
    RequestOutput,
    SamplingParams,
    Status,
)
from scalellm.downloader import download_hf_model


class OutputError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__()
        self.code = code
        self.message = message


class OutputStream:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._cancelled = False

    def put(self, item: RequestOutput) -> bool:
        if self._cancelled:
            return False

        if item.status is not None and not item.status.ok:
            self._queue.put_nowait(OutputError(item.status.code, item.status.message))
            return False

        self._queue.put_nowait(item)
        if item.finished:
            self._queue.put_nowait(StopIteration())
        return True

    # report an error to the stream, reraise as an exception
    def error(self, error: str) -> bool:
        self._queue.append(Exception(error))
        return True

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(None)

    def __iter__(self):
        return self

    def __next__(self) -> RequestOutput:
        item = self._queue.get(block=True)
        # reraise the exception
        if isinstance(item, Exception):
            raise item
        return item


class OutputAsyncStream:
    """A stream of RequestOutput objects, which can be used to
    send responses to the client asynchronously."""

    def __init__(self) -> None:
        # asyncio.Queue is used to store the items in the stream
        self._queue = asyncio.Queue()
        self._cancelled = False

    # put item into the stream
    # None to indicate the end of the stream
    def put(self, item: RequestOutput) -> bool:
        # if the stream is cancelled, return False
        if self._cancelled:
            return False

        if item.status is not None and not item.status.ok:
            self._queue.put_nowait(OutputError(item.status.code, item.status.message))
            return False

        # put the item into the queue
        self._queue.put_nowait(item)
        if item.finished:
            self._queue.put_nowait(StopAsyncIteration())
        return True

    # report an error to the stream, rerais as an exception
    def error(self, error: str) -> bool:
        self._queue.put_nowait(Exception(error))
        return True

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    # async generator to iterate over the stream
    async def __anext__(self) -> RequestOutput:
        item = await self._queue.get()
        # reraise the exception
        if isinstance(item, Exception):
            raise item
        return item


class AsyncLLMEngine:
    def __init__(self, model_path: str, devices: str):
        if not os.path.exists(model_path):
            model_path = download_hf_model(model_path)
        self._handler = LLMHandler(model_path, devices)

    # schedule a request to the engine, and return a stream to receive output
    async def schedule_async(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        priority: Priority,
        stream: bool,
    ) -> OutputAsyncStream:
        output_stream = OutputAsyncStream()

        def callback(output: RequestOutput) -> bool:
            return output_stream.put(output)

        self._handler.schedule_async(
            prompt, sampling_params, priority, stream, callback
        )
        return output_stream

    async def schedule_chat_async(
        self,
        messages: List[Message],
        sampling_params: SamplingParams,
        priority: Priority,
        stream: bool,
    ) -> OutputAsyncStream:
        output_stream = OutputAsyncStream()

        def callback(output: RequestOutput) -> bool:
            return output_stream.put(output)

        self._handler.schedule_chat_async(
            messages, sampling_params, priority, stream, callback
        )
        return output_stream

    def schedule(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        priority: Priority,
        stream: bool,
    ) -> OutputStream:
        output_stream = OutputStream()

        def callback(output: RequestOutput) -> bool:
            return output_stream.put(output)

        self._handler.schedule_async(
            prompt, sampling_params, priority, stream, callback
        )
        return output_stream

    def schedule_chat(
        self,
        messages: List[Message],
        sampling_params: SamplingParams,
        priority: Priority,
        stream: bool,
    ) -> OutputStream:
        output_stream = OutputStream()

        def callback(output: RequestOutput) -> bool:
            return output_stream.put(output)

        self._handler.schedule_chat_async(
            messages, sampling_params, priority, stream, callback
        )
        return output_stream

    # start the engine, non-blocking
    def start(self) -> None:
        return self._handler.start()

    # stop the engine, non-blocking
    def stop(self) -> None:
        return self._handler.stop()
