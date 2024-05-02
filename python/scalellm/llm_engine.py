import asyncio
from scalellm.output import RequestOutput


class RequestOutputStream:
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

    # report an error to the stream
    def error(self, error: str) -> None:
        self._queue.put_nowait(Exception(error))
        
    # cancel the stream
    def cancel(self) -> None:
      self._cancelled = True

    def __aiter__(self):
        return self

    # async generator to iterate over the stream
    async def __anext__(self) -> RequestOutput:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        # reraise the exception
        if isinstance(item, Exception):
            raise item
        return item


class LLMEngine:
    def __init__(self):
        pass

    # schedule a request to be processed, and return a RequestOutputStream
    def schedule_request(request) -> RequestOutputStream:
        stream = RequestOutputStream()
        # callback = lambda output: stream.put(output)
        #  _engine.schedule_request(request, callback)
        return stream

    # start the engine
    def start(self) -> None:
        # _engine.start()
        pass

    def stop(self) -> None:
        # _engine.stop()
        pass
