#!/usr/bin/env python

import llm

def test_llm_generate():
  sampling_parameter = llm.SamplingParameter()
  sampling_parameter.temperature = 0

  stopping_criteria = llm.StoppingCriteria()
  stopping_criteria.max_tokens = 100
  stopping_criteria.ignore_eos_token = False

  test_llm = llm.LLM("/data/llama-2-7b-hf/", sampling_parameter,
      stopping_criteria, 100, "cuda:0")
  test_llm.generate(["who is messi"])

def main():
  test_llm_generate()

if __name__ == "__main__":
  main()
