#!/usr/bin/env python3

from scalellm import LLM, SamplingParams


def test_llm_generate():
  sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    ignore_eos=True,
  )

    # Create an LLM.
  llm = LLM(model="gpt2", devices="cuda")
  # Generate texts from the prompts. The output is a list of RequestOutput objects
  # that contain the prompt, generated text, and other information.
  llm.generate(["who is messi"], sampling_params)

def main():
  test_llm_generate()

if __name__ == "__main__":
  main()
