from typing import List, Optional

import torch


def masked_self_attention(
    query: torch.Tensor,  # [q_len, n_heads, head_dim]
    key: torch.Tensor,  # [kv_len, n_heads, head_dim]
    value: torch.Tensor,  # [kv_len, n_heads, head_dim]
    alibi_bias: Optional[torch.Tensor], # [n_heads, 1, kv_len]
    mask: torch.Tensor,  # [n_heads, q_len, kv_len]
    sm_scale: float,
    logits_soft_cap: float,
) -> torch.Tensor:
    # => [n_heads, q_len, kv_len]
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float())

    # apply scale
    scores *= sm_scale

    # apply soft_cap
    if logits_soft_cap > 0.0:
        scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        
    # apply alibi bias
    if alibi_bias is not None:
        scores += alibi_bias

    # apply mask
    scores.masked_fill_(mask == 0, float("-inf"))

    # softmax => [n_heads, q_len, kv_len]
    scores = torch.softmax(scores, dim=-1)
    # => [q_len, n_heads, head_dim]
    return torch.einsum("hqk,khd->qhd", scores, value.float()).type_as(query)


def varlen_masked_self_attention(
    query: torch.Tensor,  # [q_len, n_heads, head_dim]
    key_cache: torch.Tensor,  # [n_blocks, block_size, n_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [n_blocks, block_size, n_kv_heads, head_dim]
    query_lens: List[int],  # [batch_size]
    kv_lens: List[int],  # [batch_size]
    block_tables: torch.Tensor,  # block ids for each sequence, [batch_size, max_num_blocks]
    sm_scale: float,
    logits_soft_cap: float = 0.0,
    sliding_window: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert key_cache.shape == value_cache.shape

    n_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, n_heads, head_dim = query.shape
    _, block_size, n_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    # process sequence one by one
    for i in range(n_seqs):
        q_len = query_lens[i]
        kv_len = kv_lens[i]
        assert kv_len >= q_len

        # [q_len, n_heads, head_dim]
        q = query[start_idx : start_idx + q_len]

        # retrieve key and value from cache
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        # block ids for current sequence
        block_ids = block_tables[i, :num_kv_blocks]

        # [kv_len, n_kv_heads, head_size]
        k = key_cache[block_ids].view(-1, n_kv_heads, head_size)[:kv_len]
        v = value_cache[block_ids].view(-1, n_kv_heads, head_size)[:kv_len]

        if n_heads != n_kv_heads:
            assert n_heads % n_kv_heads == 0
            n_groups = n_heads // n_kv_heads
            k = k.repeat_interleave(repeats=n_groups, dim=1)
            v = v.repeat_interleave(repeats=n_groups, dim=1)

        # create mask [1, q_len, kv_len]
        mask = torch.ones(1, q_len, kv_len).bool()
        if sliding_window >= 0:
            # returns the upper triangular part of a matrix
            mask = torch.triu(mask, diagonal=kv_len - q_len - sliding_window)

        # causal mask
        # returns the lower triangular part of a matrix
        mask = mask.tril(diagonal=kv_len - q_len).to(query)

        # calculate alibi attention bias
        alibi_bias = None
        if alibi_slopes is not None:
            assert alibi_slopes.shape == (n_heads,)
            # since it's causal mask, we can just use [0, 1, ...,, kv_len)
            distance = torch.arange(kv_len, dtype=torch.float32)
            # [n_heads, 1, kv_len]
            alibi_bias = distance.view(1, 1, -1) * alibi_slopes.view(n_heads, 1, 1)
            

        out = masked_self_attention(
            query=q,
            key=k,
            value=v,
            alibi_bias=alibi_bias,
            mask=mask,
            sm_scale=sm_scale,
            logits_soft_cap=logits_soft_cap,
        )

        outputs.append(out)
        start_idx += q_len

    return torch.cat(outputs, dim=0)
