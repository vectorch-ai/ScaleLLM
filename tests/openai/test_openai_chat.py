import sys

import pytest
import requests
from openai import AsyncClient
from openai_server import OpenAIServer


@pytest.fixture(scope="module")
def server():
    with OpenAIServer(
        [
            "--model",
            "gpt2",
            "--max_cache_size",
            1024 * 1024 * 1024, # 1GB
        ]
    ) as server:
        yield server


@pytest.fixture
def client():
    client = AsyncClient(
        base_url="http://localhost:8080/v1",
        api_key="EMPTY_API_TOKEN",
    )
    yield client
    del client


@pytest.mark.asyncio
async def test_basic(server: OpenAIServer, client: AsyncClient):
    response = requests.get("http://localhost:8080/health")
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main(sys.argv)
