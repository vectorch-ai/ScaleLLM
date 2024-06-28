#!/usr/bin/env python3

import torch
from scalellm import VLM, SamplingParameter, StoppingCriteria

def test_pixel_value_llava_generate():
  vlm = VLM(
      model="llava-hf/llava-1.5-7b-hf",
      image_input_type="pixel_values",
      image_token_id=32000,
      image_input_shape="1,3,336,336",
      image_feature_size=576,
  )

  prompt = "<image>" * 576 + (
      "\nUSER: What is the content of this image?\nASSISTANT:")

  # This should be provided by another online or offline component.
  image = torch.load("images/stop_sign_pixel_values.pt")

  output = vlm.generate(images, prompt)
  print(o.outputs[0].text)

def main():
  test_pixel_value_llava_generate()

if __name__ == "__main__":
  main()
