from typing import List, Tuple

import pytest
import torch
from ref_attention import varlen_masked_self_attention

import scalellm._C.kernels as kernels  # type: ignore


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize("seq_lens", [[(1, 100)], [(100, 100)], [(1, 100), (15, 15), (111, 234), (1000, 10000)]])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("n_blocks", [100])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
@pytest.mark.parametrize("sliding_window", [-1, 50])
@pytest.mark.parametrize("alibi", [False, True])
@torch.inference_mode
def test_flashinfer_varlen_masked_self_attention_fp8_kv(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    n_blocks: int,
    block_size: int,
    logits_soft_cap: float,
    sliding_window: int,
    alibi: bool,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    n_seqs = len(seq_lens)
    q_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]

    n_heads, n_kv_heads = num_heads
    assert n_heads % n_kv_heads == 0
    max_kv_len = max(kv_lens)
    sm_scale = head_size**-0.5

    # Generate random query, key, and value tensors.
    query = torch.randn(sum(q_lens), n_heads, head_size, dtype=dtype)
    key_cache = 0.05 * torch.randn(
        n_blocks, block_size, n_kv_heads, head_size, dtype=dtype
    )
    value_cache = 0.05 * torch.randn(
        n_blocks, block_size, n_kv_heads, head_size, dtype=dtype
    )

    max_n_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, n_blocks, (n_seqs, max_n_blocks_per_seq), dtype=torch.int32
    )

    # prepare input tensors for the kernel
    qo_indptr = [0]
    kv_indptr = [0]
    paged_kv_indptr = [0]
    paged_kv_indices = []
    for i in range(n_seqs):
        qo_indptr.append(qo_indptr[-1] + q_lens[i])
        kv_indptr.append(kv_indptr[-1] + kv_lens[i])

        seq_len = kv_lens[i]
        assert seq_len > 0

        num_blocks = (seq_len + block_size - 1) // block_size
        paged_kv_indices.extend(block_tables[i, :num_blocks])
        paged_kv_indptr.append(len(paged_kv_indices))

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32)
    paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32)

    wrapper = kernels.BatchPrefillWrapper(False)
    # TODO: determine the best size for the workspace buffer.
    float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    int_workspace_buffer = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)

    empty_q_data = torch.empty(0, dtype=dtype)

    num_sm = -1
    wrapper.plan(
        float_workspace_buffer,
        int_workspace_buffer,
        qo_indptr,
        paged_kv_indptr,
        n_seqs,
        n_heads,
        n_kv_heads,
        head_size,
        block_size,
        empty_q_data,
        num_sm,
    )

    alibi_slopes = torch.randn(n_heads, dtype=torch.float32) if alibi else None

    # The calibration scale of key/value for fp8 input
    k_scale = key_cache.amax().item() / 256.0
    v_scale = value_cache.amax().item() / 256.0

    key_cache_fp8 = (key_cache / k_scale).to(kv_dtype)
    value_cache_fp8 = (value_cache / v_scale).to(kv_dtype)

    output_fp8 = wrapper.run(
        query,
        qo_indptr,
        kv_indptr,
        key_cache_fp8,
        value_cache_fp8,
        paged_kv_indptr,
        paged_kv_indices,
        sliding_window,
        logits_soft_cap,
        sm_scale * k_scale,  # scaled key by k_scale
        alibi_slopes,
    )
    output_fp8 *= v_scale

    ref_output = varlen_masked_self_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=q_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        alibi_slopes=alibi_slopes,
    )

    torch.testing.assert_close(output_fp8, ref_output, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__])
