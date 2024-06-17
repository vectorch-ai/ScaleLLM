import sys

import openai
import pytest
from openai_server import OpenAIServer

# a small chat model for testing
MODEL_NAME = "gpt2"


@pytest.fixture(scope="class")
def server():
    with OpenAIServer(
        [
            "--model",
            MODEL_NAME,
            "--devices",
            "cpu",
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


class TestOpenAIComplete:
    @pytest.fixture(scope="class", autouse=True)
    def start_server(self, server):
        # wait up to 60 seconds for the server to start
        server.wait_ready(60)

    @pytest.mark.asyncio
    async def test_parameter_validation(self, client):
        prompt = "what is llm?"
        # unknow model
        with pytest.raises(openai.NotFoundError) as error:
            await client.completions.create(
                model="unknow_model_name",
                prompt=prompt,
            )
        assert error.value.response.status_code == 404

        # temperture out of range, valid range is [0.0, 2.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                temperature=3.0,
            )
        assert error.value.response.status_code == 400

        # top_p between [0.0, 1.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                top_p=1.1,
            )
        assert error.value.response.status_code == 400

        # logprobs between [0, 20]
        with pytest.raises(openai.BadRequestError) as error:
            await client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                logprobs=21,
            )
        assert error.value.response.status_code == 400

        # presence_penality between [-2.0, 2.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                presence_penalty=2.1,
            )
        assert error.value.response.status_code == 400

        # frequency_penality between [0.0, 2.0]
        with pytest.raises(openai.BadRequestError) as error:
            await client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                frequency_penalty=-1.0,
            )
        assert error.value.response.status_code == 400

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        models = await client.models.list()
        models = models.data
        assert len(models) == 1
        served_model = models[0]
        assert served_model.id == MODEL_NAME
        assert served_model.owned_by == "scalellm"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [1, 2, 4])
    async def test_completions(self, client, n):
        prompt = "what is llm?"
        output = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
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
            assert choice.text is not None
            assert choice.logprobs is None
            assert choice.finish_reason == "length"
        assert output.usage == openai.types.CompletionUsage(
            completion_tokens=10 * n, prompt_tokens=5, total_tokens=5 + 10 * n
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("logprobs", [1, 2, 4])
    async def test_logprobs(self, client, logprobs):
        prompt = "what is llm?"
        output = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=10,
            logprobs=logprobs,
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
            assert choice.text is not None
            assert choice.logprobs is not None
            
            assert len(choice.logprobs.tokens) == 10
            assert len(choice.logprobs.token_logprobs) == 10
            assert len(choice.logprobs.top_logprobs) == 10
            for top_logprob in choice.logprobs.top_logprobs:
                assert len(top_logprob) == logprobs

    @pytest.mark.asyncio
    async def test_logprobs_streaming(self, client):
        prompt = "what is llm?"
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            logprobs=5,
            stream=False,
        )
        ref_tokens = completion.choices[0].logprobs.tokens
        ref_token_logprobs = completion.choices[0].logprobs.token_logprobs
        ref_top_logprobs = completion.choices[0].logprobs.top_logprobs
        ref_finish_reason = completion.choices[0].finish_reason

        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            logprobs=5,
            stream=True,
        )

        tokens = []
        token_logprobs = []
        top_logprobs = []
        finish_reason = None
        async for chunk in stream:
            assert len(chunk.choices) == 1
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if choice.logprobs:
                tokens.extend(choice.logprobs.tokens)
                token_logprobs.extend(choice.logprobs.token_logprobs)
                top_logprobs.extend(choice.logprobs.top_logprobs)

        assert finish_reason == ref_finish_reason
        assert tokens == ref_tokens
        assert token_logprobs == ref_token_logprobs
        assert top_logprobs == ref_top_logprobs


if __name__ == "__main__":
    pytest.main(sys.argv)
