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


def test_logprobs():
    with LLM(model="gpt2", devices="cpu") as llm:
        outputs = llm.generate(
            ["who is messi", "how to use pytest", "hello,", "what is llm"],
            [
                SamplingParams(
                    temperature=0,
                    max_tokens=35,
                    ignore_eos=True,
                    logprobs=True,
                    top_logprobs=10,
                ),
                SamplingParams(
                    temperature=0,
                    max_tokens=64,
                    ignore_eos=True,
                    logprobs=True,
                    top_logprobs=20,
                ),
                SamplingParams(
                    temperature=0.5, max_tokens=35, ignore_eos=True, logprobs=True
                ),
                SamplingParams(temperature=0.4, max_tokens=35, ignore_eos=True),
            ],
        )
        assert len(outputs) == 4
        output = outputs[0]
        assert output.finished
        assert output.prompt == "who is messi"
        assert output.usage
        assert output.outputs[0].text
        assert output.outputs[0].index == 0
        assert output.outputs[0].finish_reason == "length"

        # Check logprobs
        generated_tokens = output.usage.num_generated_tokens
        token_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs
        assert len(logprobs) == len(token_ids)
        assert generated_tokens == len(logprobs)
        for token_id, logprob in zip(token_ids, logprobs):
            assert logprob.token_id == token_id
            top_logprobs = logprob.top_logprobs
            assert len(top_logprobs) == 10
            assert logprob.token == top_logprobs[0].token
            assert logprob.token_id == top_logprobs[0].token_id
            assert logprob.logprob == top_logprobs[0].logprob

        output = outputs[1]
        assert output.finished
        assert output.prompt == "how to use pytest"
        assert output.usage
        assert output.outputs[0].text
        assert output.outputs[0].index == 0
        assert output.outputs[0].finish_reason == "length"

        # Check logprobs
        generated_tokens = output.usage.num_generated_tokens
        token_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs
        assert len(logprobs) == len(token_ids)
        assert generated_tokens == len(logprobs)
        for token_id, logprob in zip(token_ids, logprobs):
            assert logprob.token_id == token_id
            top_logprobs = logprob.top_logprobs
            assert len(top_logprobs) == 20
            assert logprob.token == top_logprobs[0].token
            assert logprob.token_id == top_logprobs[0].token_id
            assert logprob.logprob == top_logprobs[0].logprob

        output = outputs[2]
        assert output.finished
        assert output.prompt == "hello,"
        assert output.usage
        assert output.outputs[0].text
        assert output.outputs[0].index == 0
        assert output.outputs[0].finish_reason == "length"

        # Check logprobs
        generated_tokens = output.usage.num_generated_tokens
        token_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs
        assert len(logprobs) == len(token_ids)
        assert generated_tokens == len(logprobs)
        for token_id, logprob in zip(token_ids, logprobs):
            assert logprob.token_id == token_id
            assert logprob.top_logprobs is None

        output = outputs[3]
        assert output.finished
        assert output.prompt == "what is llm"
        assert output.usage
        assert output.outputs[0].text
        assert output.outputs[0].index == 0
        assert output.outputs[0].finish_reason == "length"
        assert output.outputs[0].logprobs is None


def test_best_of():
    with LLM(model="gpt2", devices="cpu") as llm:
        outputs = llm.generate(
            ["hello,"],
            SamplingParams(
                temperature=0.5, max_tokens=35, ignore_eos=True, n=2, best_of=4
            ),
        )

        assert len(outputs) == 1
        output = outputs[0]
        assert len(output.outputs) == 2

        assert output.outputs[0].text
        assert output.outputs[0].index == 0
        assert output.outputs[0].finish_reason == "length"
        assert output.outputs[0].logprobs is None

        assert output.outputs[1].text
        assert output.outputs[1].index == 1
        assert output.outputs[1].finish_reason == "length"
        assert output.outputs[1].logprobs is None
