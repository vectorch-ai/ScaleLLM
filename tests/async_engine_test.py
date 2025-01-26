import sys

import pytest

from scalellm import AsyncLLMEngine, SamplingParams


@pytest.fixture(scope="module")
def engine():
    with AsyncLLMEngine(model="gpt2", devices="cuda") as engine:
        yield engine

def test_stream_output(engine: AsyncLLMEngine):
    sampling_params = SamplingParams(temperature=0, max_tokens=100, echo=True)

    output_stream = engine.schedule(
        prompt="今天真的很热",
        sampling_params=sampling_params,
        stream=True,
    )
    stream_output_text = ""
    stream_output_token_ids = []
    for output in output_stream:
        if len(output.outputs) > 0:
            stream_output_text += output.outputs[0].text
            stream_output_token_ids.extend(output.outputs[0].token_ids)

    output_stream = engine.schedule(
        prompt="今天真的很热",
        sampling_params=sampling_params,
        stream=False,
    )
    output_text = None
    output_token_ids = None
    output = output_stream.__next__()
    if len(output.outputs) > 0:
        output_text = output.outputs[0].text
        output_token_ids = output.outputs[0].token_ids
    assert stream_output_text == output_text
    assert stream_output_token_ids == output_token_ids


def test_context_manager(engine: AsyncLLMEngine):
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        ignore_eos=True,
    )
    output_stream = engine.schedule(
        prompt="who is messi",
        sampling_params=sampling_params,
        stream=False,
    )

    output = output_stream.__next__()
    assert output.outputs[0].text
    assert output.outputs[0].index == 0
    assert output.outputs[0].finish_reason == "length"
    assert output.finished
    assert output.prompt == "who is messi"
    assert output.usage


if __name__ == "__main__":
    pytest.main(sys.argv)
