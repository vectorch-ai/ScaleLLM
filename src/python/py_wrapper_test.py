#!/usr/bin/env python

import gen_py_wrappers as wrapper

class LLM:
  """
  Use to wrapper c++ LLM
  """
  def __init__(
      self,
      devices: str
  ) -> None:
      self.devices = devices
      self.llm = wrapper.LLM(devices)

llm = wrapper.LLM("")
print(llm)
