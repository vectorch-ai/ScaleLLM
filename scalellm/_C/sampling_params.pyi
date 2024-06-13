from typing import List, Optional

# Defined in csrc/sampling_params.cpp
class SamplingParams:
    def __init__(
        self,
        max_tokens: int = 16,
        n: int = 1,
        best_of: Optional[int] = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        logprobs: bool = False,
        top_logprobs: int = 0,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    # number of tokens to generate. truncted to model's max context length.
    max_tokens: int
    # number of sequences to generate for each prompt.
    n: int
    # whether to include the original prompt in the completion response.
    echo: bool
    # frequency penalty to reduce the likelihood of generating the same word multiple times. values between [0.0, 2.0].
    frequency_penalty: float
    # presence penalty to reduce the likelihood of generating words already in the prompt. values between [-2.0, 2.0].
    presence_penalty: float
    # repetition penalty to penalize new tokens based on their occurence in the text.
    repetition_penalty: float
    # temperature of the sampling, between [0, 2].
    temperature: float
    # top_p sampling cutoff, between [0.0, 1.0].
    top_p: float
    # top_k sampling cutoff. default = 0 to disable.
    top_k: int
    # Whether to return log probabilities of the output tokens or not.
    logprobs: bool
    # An integer between 0 and 20 specifying the number of most likely tokens to return at each token position.
    top_logprobs: int
    #  ############ stopping criterias. ############
    # whether to skip special tokens in the output text. default = true.
    skip_special_tokens: bool
    # whether to ignore eos token when checking stopping criterias. default = false.
    ignore_eos: bool
    # the list of strings to stop generating further tokens.
    stop: Optional[List[str]]
    # the list of token ids to stop generating further tokens.
    stop_token_ids: Optional[List[int]]
