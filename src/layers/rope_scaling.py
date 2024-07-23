import torch
import math


def compute_default_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    return inv_freq


def apply_llama3_rope_scaling(
    inv_freq: torch.Tensor,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: int,
) -> torch.Tensor:
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)


if __name__ == "__main__":
    rotary_dim = 128
    theta = 500000.0
    inv_freq = compute_default_inv_freq(rotary_dim, theta)
    print(inv_freq)

    # apply rope scaling
    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192
    new_freqs = apply_llama3_rope_scaling(
        inv_freq, factor, low_freq_factor, high_freq_factor, old_context_len
    )
    print(new_freqs)
