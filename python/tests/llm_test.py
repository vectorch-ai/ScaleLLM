#!/usr/bin/env python3

from scalellm import LLM, SamplingParams


def test_llm_generate():
    sampling_params = SamplingParams(
        temperature=0, top_p=1.0, max_tokens=100, echo=True
    )

    test_llm = LLM(model="/data/llama-2-7b-hf/", devices="cuda")
    test_llm.generate(["who is messi"], sampling_params)


def main():
    test_llm_generate()


if __name__ == "__main__":
    main()
