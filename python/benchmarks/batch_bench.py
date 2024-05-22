import argparse
import time
import math
import json
import torch

from scalellm import LLM, SamplingParams

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_dir', type=str)
    return parser.parse_args(args=args)

def read_json_input(input_file=None):
    input_prompt = []
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
            for idx, obj in enumerate(data):
                prompt = obj['prompt']
                input_prompt.append(prompt)
    return input_prompt           

def decode_prompt(batch_input_prompt,
                  tokenizer,
                  add_special_tokens = True,
                  max_input_length = 923):
    batch_input_ids = []
    for prompt in batch_input_prompt:
        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens = add_special_tokens,
            truncation = True,
            max_length=max_input_length)
        batch_input_ids.append(input_ids)
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids

def print_output(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def main(args):
    input_prompt = read_json_input(input_file=args.input_file)
    loop_count = math.ceil(len(input_prompt) / args.batch_size)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    #sampling_params = SamplingParams(temperature=1.0, top_p=0.01, max_tokens=100)
    llm = LLM(model=args.model_dir, max_num_seqs=args.batch_size, trust_remote_code=True, enforce_eager=True)
    total_time_cost = 0

    for i in range(0, loop_count):
        batched_input_prompt = input_prompt[i*args.batch_size:(i+1)*args.batch_size]
        t1 = time.time()
        outputs = llm.generate(batched_input_prompt, sampling_params, use_tqdm=False)
        t2 = time.time()
        total_time_cost += t2-t1
        
        # total_length = 0
        # for output in outputs:
        #     total_length += len(output.outputs[0].token_ids)
        
        print("current time cost:%ss:" % (t2-t1))
        # print("avg length: %s" % (total_length // args.batch_size))
        #print_output(outputs)
    print("=================Time Consumption================")
    print("total time cost: %ss:" % total_time_cost)
    print("average time cost: %ss:" % (total_time_cost / loop_count))
    print("=================================================")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)