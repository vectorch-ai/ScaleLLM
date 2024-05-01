from scalellm import wrapper


class SamplingParameter:
  """
  Used to wrapper c++ SamplingParameter
  """
  def __init__(self):
    self.sampling_parameter = wrapper.SamplingParameter()

  def get_data(self):
    return self.sampling_parameter

  @property
  def frequency_penalty(self):
    return self.sampling_parameter.frequency_penalty

  @frequency_penalty.setter
  def frequency_penalty(self, new_frequency_penalty):
    self.sampling_parameter.frequency_penalty = new_frequency_penalty

  @property
  def presence_penalty(self):
    return self.sampling_parameter.presence_penalty

  @presence_penalty.setter
  def presence_penalty(self, new_presence_penalty):
    self.sampling_parameter.presence_penalty = new_presence_penalty

  @property
  def repetition_penalty(self):
    return self.sampling_parameter.repetition_penalty

  @repetition_penalty.setter
  def repetition_penalty(self, new_repetition_penalty):
    self.sampling_parameter.repetition_penalty = new_repetition_penalty

  @property
  def temperature(self):
    return self.sampling_parameter.temperature

  @temperature.setter
  def temperature(self, new_temperature):
    self.sampling_parameter.temperature = new_temperature

  @property
  def top_p(self):
    return self.sampling_parameter.top_p

  @top_p.setter
  def top_p(self, new_top_p):
    self.sampling_parameter.top_p = new_top_p

  @property
  def top_k(self):
    return self.sampling_parameter.top_k

  @top_k.setter
  def top_k(self, new_top_k):
    self.sampling_parameter.top_k = new_top_k

  @property
  def do_sample(self):
    return self.sampling_parameter.do_sample

  @do_sample.setter
  def do_sample(self, new_do_sample):
    self.sampling_parameter.do_sample = new_do_sample

  @property
  def seed(self):
    return self.sampling_parameter.seed

  @seed.setter
  def seed(self, new_seed):
    self.sampling_parameter.seed = new_seed

class StoppingCriteria:
  """
  Used to wrapper c++ StoppingCriteria
  """
  def __init__(self):
    self.stopping_criteria = wrapper.StoppingCriteria()

  def get_data(self):
    return self.stopping_criteria

  @property
  def max_tokens(self):
    return self.stopping_criteria.max_tokens

  @max_tokens.setter
  def max_tokens(self, new_max_tokens):
    self.stopping_criteria.max_tokens = new_max_tokens

  @property
  def eos_token_id(self):
    return self.stopping_criteria.eos_token_id

  @eos_token_id.setter
  def eos_token_id(self, new_eos_token_id):
    self.stopping_criteria.eos_token_id = new_eos_token_id

  @property
  def ignore_eos_token(self):
    return self.stopping_criteria.ignore_eos_token

  @ignore_eos_token.setter
  def ignore_eos_token(self, new_ignore_eos_token):
    self.stopping_criteria.ignore_eos_token = new_ignore_eos_token

  @property
  def stop_token_ids(self):
    return self.stopping_criteria.stop_token_ids

  @stop_token_ids.setter
  def stop_token_ids(self, new_stop_token_ids):
    self.stopping_criteria.stop_token_ids = new_stop_token_ids

  @property
  def stop_sequences(self):
    return self.stopping_criteria.stop_sequences

  @stop_sequences.setter
  def stop_sequences(self, new_stop_sequences):
    self.stopping_criteria.stop_sequences = new_stop_sequences

class LLM:
  """
  Used to wrapper c++ LLM
  """
  def __init__(
      self,
      model_path: str,
      sampling_parameter: SamplingParameter,
      stopping_criteria: StoppingCriteria,
      max_seq_len: int,
      devices: str
  ) -> None:
    self.sampling_parameter = sampling_parameter
    self.stopping_criteria = stopping_criteria
    self.llm = wrapper.LLM(model_path,
                           self.sampling_parameter.get_data(),
                           self.stopping_criteria.get_data(),
                           max_seq_len,
                           devices)

  def generate(self, batched_prompt: list[str]) -> None:
    self.llm.generate(batched_prompt)
