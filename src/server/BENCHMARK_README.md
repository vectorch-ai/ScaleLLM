## Run Offline Batched Benchmark

### Build the source file

offline batched benchmark file is in src/server/benchmark_test.cpp

test dataset is in dataset/Chatbot_group_10_100.json

### Run benchmark

benchmark_test --model_name_or_path /data/llama-2-7b-hf/ --input_file dataset/Chatbot_group_10_100.json --batch_size 8


### Run benchmark with nsys

nsys profile --stats=true --sample=cpu --capture-range=cudaProfilerApi --capture-range-end=stop -x true \
--trace=cuda,cudnn,cublas,nvtx,osrt,oshmem --cudabacktrace=kernel:1000000,sync:1000000,memory:1000000 \
--sampling-period=1000000 --delay=10 --duration=60 --wait=all  ./build2/src/server/benchmark_test \
--model_name_or_path /data/llama-2-7b-hf/ --input_file /data/dataset/Chatbot_group_10_100.json --batch_size 8

### Test result

Use scalellm with batchsize=8, would get following result (seconds):

request cost:16.4369
one_batch_cost:16.4389
average cost:16.4991
