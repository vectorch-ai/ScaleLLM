import requests
import pytest
from openai_server import OpenAIServer


@pytest.fixture(scope="module")
def openai_server():
    with OpenAIServer(
        [
            "--model",
            "gpt2",
            "--max_cache_size",
            1024 * 1024 * 1024,
        ]
    ) as server:
        yield server


def test_basic(openai_server):
    response = requests.get("http://localhost:8080/health")
    assert response.status_code == 200


if __name__ == "__main__":
    with OpenAIServer([]) as server:
        test_basic(server)
