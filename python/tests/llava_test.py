#!/usr/bin/env python3

import torch
from scalellm import LLM, SamplingParameter, StoppingCriteria

def test_pixel_value_llava_generate():
  llm = LLM(
      model="llava-hf/llava-1.5-7b-hf",
      image_input_type="pixel_values",
      image_token_id=32000,
      image_input_shape="1,3,336,336",
      image_feature_size=576,
  )

  prompt = "<image>" * 576 + (
      "\nUSER: What is the content of this image?\nASSISTANT:")

  # This should be provided by another online or offline component.
  images = torch.load("images/stop_sign_pixel_values.pt")

  outputs = llm.generate(prompt,
                         multi_modal_data=MultiModalData(
                             type=MultiModalData.Type.IMAGE, data=images))
  for o in outputs:
      generated_text = o.outputs[0].text
      print(generated_text)

def test_image_feature_llava_generate():
  llm = LLM(
      model="llava-hf/llava-1.5-7b-hf",
      image_input_type="image_features",
      image_token_id=32000,
      image_input_shape="1,576,1024",
      image_feature_size=576,
  )

  prompt = "<image>" * 576 + (
      "\nUSER: What is the content of this image?\nASSISTANT:")

  # This should be provided by another online or offline component.
  images = torch.load("images/stop_sign_image_features.pt")

  outputs = llm.generate(prompt,
                         multi_modal_data=MultiModalData(
                             type=MultiModalData.Type.IMAGE, data=images))
  for o in outputs:
      generated_text = o.outputs[0].text
      print(generated_text)

def main():
  test_image_feature_llava_generate()
  test_pixel_value_llava_generate()

if __name__ == "__main__":
  main()
