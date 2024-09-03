import asyncio
import os
import queue
from typing import List, Optional

from scalellm._C import (LLMHandler, Message, Priority, RequestOutput,
                         SamplingParams)
from scalellm.downloader import download_hf_model
from scalellm.errors import ValidationError


class OutputStream:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._cancelled = False

    def put(self, item: RequestOutput) -> bool:
        if self._cancelled:
            return False

        if item.status is not None and not item.status.ok:
            self._queue.put_nowait(
                ValidationError(item.status.code, item.status.message)
            )
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
        self._queue.put_nowait(StopIteration())

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
        # asyncio.Queue is used to store the items in the stream, not thread-safe
        self._queue = asyncio.Queue()
        # event loop used to schedule callbacks from other threads
        self._loop = asyncio.get_running_loop()
        assert self._loop is not None
        self._cancelled = False

    def _put_nowait(self, item):
        # put item into asyncio.queue in a thread-safe way
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    # put item into the stream
    # None to indicate the end of the stream
    def put(self, item: RequestOutput) -> bool:
        # if the stream is cancelled, return False
        if self._cancelled:
            return False

        if item.status is not None and not item.status.ok:
            self._put_nowait(ValidationError(item.status.code, item.status.message))
            return False

        # put the item into the queue
        self._put_nowait(item)
        if item.finished:
            self._put_nowait(StopAsyncIteration())
        return True

    # report an error to the stream, rerais as an exception
    def error(self, error: str) -> bool:
        self._queue.put_nowait(Exception(error))
        return True

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(StopAsyncIteration())

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
    def __init__(
        self,
        model: str,
        revision: Optional[str] = None,
        draft_model: Optional[str] = None,
        draft_revision: Optional[str] = None,
        allow_patterns: Optional[str] = None,
        cache_dir: Optional[str] = None,
        convert_to_safetensors: bool = False,
        devices: Optional[str] = None,
        draft_devices: Optional[str] = None,
        block_size: int = 16,
        max_cache_size: int = 0,  # 0 means that cache size is caculated by available memory
        max_memory_utilization: float = 0.9,
        enable_prefix_cache: bool = True,
        enable_cuda_graph: bool = True,
        cuda_graph_max_seq_len: int = 2048,
        cuda_graph_batch_sizes: Optional[List[int]] = None,
        draft_cuda_graph_batch_sizes: Optional[List[int]] = None,
        max_tokens_per_batch: int = 512,
        max_seqs_per_batch: int = 128,
        num_speculative_tokens: int = 0,
        num_handling_threads: int = 4,
    ) -> None:
        self._model = model
        self._draft_model = draft_model
        # download hf model if it does not exist
        model_path = model
        if not os.path.exists(model_path):
            model_path = download_hf_model(
                repo_id=model_path,
                revision=revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                convert_to_safetensors=convert_to_safetensors,
            )
        draft_model_path = draft_model
        if draft_model_path is not None and not os.path.exists(draft_model_path):
            draft_model_path = download_hf_model(
                repo_id=draft_model_path,
                revision=draft_revision,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                convert_to_safetensors=convert_to_safetensors,
            )

        options = LLMHandler.Options()
        options.model_path = model_path
        options.devices = devices
        options.draft_model_path = draft_model_path
        options.draft_devices = draft_devices
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.enable_cuda_graph = enable_cuda_graph
        options.cuda_graph_max_seq_len = cuda_graph_max_seq_len
        options.cuda_graph_batch_sizes = cuda_graph_batch_sizes
        options.draft_cuda_graph_batch_sizes = draft_cuda_graph_batch_sizes
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.num_speculative_tokens = num_speculative_tokens
        options.num_handling_threads = num_handling_threads
        # create the LLM handler
        self._handler = LLMHandler(options)

    # schedule a request to the engine, and return a stream to receive output
    async def schedule_async(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: Priority = Priority.NORMAL,
        stream: bool = False,
    ) -> OutputAsyncStream:
        output_stream = OutputAsyncStream()

        def callback(output: RequestOutput) -> bool:
            output.prompt = prompt
            return output_stream.put(output)

        # use default sampling parameters if not provided
        sampling_params = sampling_params or SamplingParams()
        self._handler.schedule_async(
            prompt, sampling_params, priority, stream, callback
        )
        return output_stream

    async def schedule_chat_async(
        self,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        priority: Priority = Priority.NORMAL,
        stream: bool = False,
    ) -> OutputAsyncStream:
        output_stream = OutputAsyncStream()

        def callback(output: RequestOutput) -> bool:
            return output_stream.put(output)

        # use default sampling parameters if not provided
        sampling_params = sampling_params or SamplingParams()
        self._handler.schedule_chat_async(
            messages, sampling_params, priority, stream, callback
        )
        return output_stream

    def schedule(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        priority: Priority = Priority.NORMAL,
        stream: bool = False,
    ) -> OutputStream:
        output_stream = OutputStream()

        def callback(output: RequestOutput) -> bool:
            output.prompt = prompt
            return output_stream.put(output)

        # use default sampling parameters if not provided
        sampling_params = sampling_params or SamplingParams()
        self._handler.schedule_async(
            prompt, sampling_params, priority, stream, callback
        )
        return output_stream

    def schedule_chat(
        self,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        priority: Priority = Priority.NORMAL,
        stream: bool = False,
    ) -> OutputStream:
        output_stream = OutputStream()

        def callback(output: RequestOutput) -> bool:
            return output_stream.put(output)

        # use default sampling parameters if not provided
        sampling_params = sampling_params or SamplingParams()
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

    def apply_chat_template(self, messages: List[Message]) -> Optional[str]:
        return self._handler.apply_chat_template(messages)

    def encode(self, text: str) -> List[int]:
        return self._handler.encode(text)

    def decode(
        self, tokens: List[int], skip_special_tokens: bool = True
    ) -> Optional[str]:
        return self._handler.decode(tokens, skip_special_tokens)

    def __del__(self):
        if hasattr(self, "_handler"):
            self._handler.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        self.__del__()
        return False

    def __repr__(self) -> str:
        if self._draft_model:
            return (
                f"AsyncLLMEngine(model={self._model}, draft_model={self._draft_model})"
            )
        return f"AsyncLLMEngine(model={self._model})"
