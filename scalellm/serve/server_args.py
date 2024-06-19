import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8080, help="port number")
    parser.add_argument(
        "--log_level", type=str, default="info", help="uvicorn log level"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="The model name used in the API.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="model name or path.",
    )

    parser.add_argument(
        "--revision", type=str, default=None, help="Revision of the model."
    )
    parser.add_argument("--devices", type=str, default="auto", help="devices to use.")
    parser.add_argument(
        "--draft_model", type=str, default=None, help="Draft model name or path."
    )
    parser.add_argument(
        "--draft_revision", type=str, default=None, help="Revision of the draft model."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="HF Model cache directory."
    )
    parser.add_argument(
        "--convert_to_safetensors", type=bool, default=False, help="Convert to SafeTensors."
    )
    parser.add_argument(
        "--draft_devices", type=str, default="auto", help="Draft devices to use."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Number of slots per kv cache block. Default is 16.",
    )
    parser.add_argument(
        "--max_cache_size",
        type=int,
        default=0,
        help="Max gpu memory size for kv cache. Default is 0, which means cache size is caculated by available memory.",
    )
    parser.add_argument(
        "--max_memory_utilization",
        type=float,
        default=0.9,
        help="The fraction of GPU memory to be used for model inference, including model weights and kv cache.",
    )
    parser.add_argument(
        "--enable_prefix_cache",
        type=lambda s: s.lower() in ["true", "t", "yes", "1"],
        default=True,
        help="Enable prefix cache.",
    )
    parser.add_argument(
        "--enable_cuda_graph",
        type=lambda s: s.lower() in ["true", "t", "yes", "1"],
        default=True,
        help="Enable CUDA graph.",
    )
    parser.add_argument(
        "--cuda_graph_max_seq_len",
        type=int,
        default=2048,
        help="Max sequence length to capture for each CUDA graph. Batches with larger values will be executed in eager mode.",
    )
    parser.add_argument(
        "--cuda_graph_batch_sizes",
        type=str,
        default=None,
        help="Batch sizes to capture for CUDA graph. Values are a list of integers separated by comma. (e.g., 1,2,4,8,16,32,64,128)",
    )
    parser.add_argument(
        "--draft_cuda_graph_batch_sizes",
        type=str,
        default=None,
        help="Batch sizes to capture for draft CUDA graph. Values are a list of integers separated by comma. (e.g., 1,2,4,8,16,32,64,128)",
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=512,
        help="Max number of tokens per batch.",
    )
    parser.add_argument(
        "--max_seqs_per_batch",
        type=int,
        default=128,
        help="Max number of sequences per batch.",
    )
    parser.add_argument(
        "--num_speculative_tokens",
        type=int,
        default=0,
        help="Number of speculative tokens.",
    )
    parser.add_argument(
        "--num_handling_threads",
        type=int,
        default=4,
        help="Number of handling threads.",
    )
    parser.add_argument("--ssl-keyfile",
                        type=str, 
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str, 
                        default=None,
                        help="The file path to the SSL cert file")
    return parser.parse_args()
