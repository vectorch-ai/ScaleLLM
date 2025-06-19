import sys

import openai
import pytest
from openai_server import OpenAIServer

# a small chat model for testing
MODEL_NAME = "JackFram/llama-68m"


@pytest.fixture(scope="class")
def server():
    with OpenAIServer(
        [
            "--model",
            MODEL_NAME,
            "--devices",
            "cuda",
            "--max_cache_size",
            1024 * 1024 * 1024,  # 1GB
            "--convert_to_safetensors=True",
        ]
    ) as server:
        yield server


@pytest.fixture(scope="class")
def client():
    client = openai.AsyncClient(
        base_url="http://localhost:8080/v1",
        api_key="EMPTY_API_TOKEN",
    )
    yield client
    del client


class TestOpenAIChat:
    @pytest.fixture(scope="class", autouse=True)
    def start_server(self, server):
        # wait up to 60 seconds for the server to start
        server.wait_ready(60)

    @pytest.mark.asyncio
    async def test_parameter_validation(self, client):
        messages = [
            {"role": "user", "content": "hi"},
        ]
        # unknow model
        with pytest.raises(openai.NotFoundError) as error:
            await client.chat.completions.create(
                model="unknow_model_name",
                messages=messages,
            )
        assert error.value.response.status_code == 404

        # temperture out of range, valid range is [0.0, 2.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=3.0,
            )
        assert error.value.response.status_code == 400

        # top_p between [0.0, 1.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                top_p=1.1,
            )
        assert error.value.response.status_code == 400

        # top_logprobs between [0, 20]
        with pytest.raises(openai.UnprocessableEntityError) as error:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                logprobs=True,
                top_logprobs=21,
            )
        assert error.value.response.status_code == 422

        # presence_penality between [-2.0, 2.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                presence_penalty=2.1,
            )
        assert error.value.response.status_code == 400

        # frequency_penality between [0.0, 2.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                frequency_penalty=-1.0,
            )
        assert error.value.response.status_code == 400

    # TODO: fix failures on 5090
    # @pytest.mark.asyncio
    # async def test_list_models(self, client):
    #     models = await client.models.list()
    #     models = models.data
    #     print("models: ", models)
    #     assert len(models) == 1
    #     served_model = models[0]
    #     assert served_model.id == MODEL_NAME
    #     assert served_model.owned_by == "scalellm"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 4])
    async def test_chat(self, client, n):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is llm?"},
        ]
        output = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            n=n,
            stream=False,
        )

        assert output.created is not None
        assert output.id is not None
        assert output.choices is not None
        assert output.usage is not None
        assert output.model == MODEL_NAME
        assert len(output.choices) == n
        for choice in output.choices:
            assert choice.index is not None
            assert choice.message is not None
            assert choice.logprobs is None
            assert choice.finish_reason == "length"

            message = choice.message
            assert message.role == "assistant"
            assert message.content is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 4])
    async def test_multi_turn_chat(self, client, n):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is llm?"},
            {"role": "assistant", "content": "llm is a large language model"},
            {"role": "user", "content": "what is the largest language model?"},
        ]
        output = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            n=n,
            stream=False,
        )

        assert output.created is not None
        assert output.id is not None
        assert output.choices is not None
        assert output.usage is not None
        assert output.model == MODEL_NAME
        assert len(output.choices) == n
        for choice in output.choices:
            assert choice.index is not None
            assert choice.message is not None
            assert choice.logprobs is None

            message = choice.message
            assert message.role == "assistant"
            assert message.content is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("logprobs", [1, 2, 4])
    async def test_logprobs(self, client, logprobs):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is llm?"},
        ]
        output = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            logprobs=True,
            top_logprobs=logprobs,
            stream=False,
        )

        assert output.created is not None
        assert output.id is not None
        assert output.choices is not None
        assert output.usage is not None
        assert output.model == MODEL_NAME
        assert len(output.choices) == 1
        for choice in output.choices:
            assert choice.index is not None
            assert choice.message is not None
            assert choice.logprobs is not None

            content = choice.logprobs.content
            assert content is not None
            for logprob in content:
                assert logprob.token is not None
                assert logprob.logprob is not None
                assert len(logprob.top_logprobs) == logprobs

    # test stream
    @pytest.mark.asyncio
    async def test_one_turn_chat_streaming(self, client):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is llm?"},
        ]

        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
        )
        ref_content = completion.choices[0].message.content
        ref_finish_reason = completion.choices[0].finish_reason

        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=True,
        )
        content = ""
        finish_reason = None
        async for chunk in stream:
            assert len(chunk.choices) == 1
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.role:
                assert delta.role == "assistant"
            if delta.content:
                content += delta.content

        # stream should return the same content and finish_reason
        assert content == ref_content
        assert finish_reason == ref_finish_reason

    @pytest.mark.asyncio
    async def test_logprobs_streaming(self, client):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is llm?"},
        ]
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
            stream=False,
        )
        ref_logprobs = completion.choices[0].logprobs.content
        ref_finish_reason = completion.choices[0].finish_reason

        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
            stream=True,
        )

        logprobs = []
        finish_reason = None
        async for chunk in stream:
            assert len(chunk.choices) == 1
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if choice.logprobs:
                content = choice.logprobs.content
                logprobs.extend(content)

        assert len(logprobs) == len(ref_logprobs)
        assert finish_reason == ref_finish_reason
        for logprob, ref_logprob in zip(logprobs, ref_logprobs):
            assert logprob.token == ref_logprob.token
            assert round(logprob.logprob, 4) == round(ref_logprob.logprob, 4)
            assert len(logprob.top_logprobs) == len(ref_logprob.top_logprobs)
            for top_logprob, ref_top_logprob in zip(
                logprob.top_logprobs, ref_logprob.top_logprobs
            ):
                assert top_logprob.token == ref_top_logprob.token
                assert round(top_logprob.logprob, 4) == round(
                    ref_top_logprob.logprob, 4
                )

    @pytest.mark.asyncio
    async def test_usage_streaming(self, client):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is llm?"},
        ]
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
        )
        ref_usage = completion.usage

        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        async for chunk in stream:
            if chunk.usage:
                assert usage is None
                usage = chunk.usage
        assert usage == ref_usage


if __name__ == "__main__":
    pytest.main(sys.argv)
