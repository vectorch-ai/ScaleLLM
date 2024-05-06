import asyncio
import queue

from scalellm._C import RequestOutput, SamplingParams, LLMHandler


class OutputStream:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._cancelled = False

    def put(self, item: RequestOutput) -> bool:
        if self._cancelled:
            return False
        self._queue.put_nowait(item)
        # put a sentinel value to indicate the end of the stream
        if item.finished:
            self._queue.put_nowait(None)
        return True

    # report an error to the stream, rerais as an exception
    def error(self, error: str) -> None:
        self._queue.append(Exception(error))

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(None)

    def __iter__(self):
        return self

    def __next__(self) -> RequestOutput:
        item = self._queue.get(block=True)
        # None indicates the end of the stream
        if item is None:
            raise StopIteration
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
        # put the item into the queue
        self._queue.put_nowait(item)
        # put a sentinel value to indicate the end of the stream
        if item.finished:
            self._queue.put_nowait(None)
        return True

    # report an error to the stream, rerais as an exception
    def error(self, error: str) -> None:
        self._queue.put_nowait(Exception(error))

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    # async generator to iterate over the stream
    async def __anext__(self) -> RequestOutput:
        item = await self._queue.get()
        # None indicates the end of the stream
        if item is None:
            raise StopAsyncIteration
        # reraise the exception
        if isinstance(item, Exception):
            raise item
        return item


class LLMEngine:
    def __init__(self, model_path: str, devices: str):
        self._handler = LLMHandler(model_path, devices)

    # schedule a request to the engine, and return a stream to receive output
    async def schedule_async(
        self, prompt: str, sampling_params: SamplingParams
    ) -> OutputAsyncStream:
        # creat a async stream to receive output
        stream = OutputAsyncStream()

        # define callback to add output to the stream
        def callback(output: RequestOutput) -> bool:
            return stream.put(output)

        # schedule the request
        self._handler.schedule(prompt, sampling_params, callback)
        return stream

    def schedule(self, prompt: str, sampling_params: SamplingParams) -> OutputStream:
        # create a stream to reeive output
        stream = OutputStream()

        def callback(output: RequestOutput):
            return stream.put(output)

        self._handler.schedule(prompt, sampling_params, callback)
        return stream

    # stop the engine
    def stop(self) -> None:
        return self._handler.stop()
