from scalellm import LLM, SamplingParams


def test_llm_generate():
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=100,
        ignore_eos=True,
    )

    # Create an LLM.
    llm = LLM(model="gpt2", devices="cpu")
    outputs = llm.generate(["who is messi"], sampling_params)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.outputs[0].text
    assert output.outputs[0].index == 0
    assert output.outputs[0].finish_reason == "length"
    assert output.finished
    assert output.prompt == "who is messi"
    assert output.usage
    del llm


def test_context_manager():
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=100,
        ignore_eos=True,
    )
    with LLM(model="gpt2", devices="cpu") as llm:
        outputs = llm.generate(["who is messi"], sampling_params)
        assert len(outputs) == 1
        output = outputs[0]
        assert output.outputs[0].text
        assert output.outputs[0].index == 0
        assert output.outputs[0].finish_reason == "length"
        assert output.finished
        assert output.prompt == "who is messi"
        assert output.usage
