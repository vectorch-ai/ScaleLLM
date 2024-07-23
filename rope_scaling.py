import torch
import math


def compute_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    return inv_freq


def apply_scaling(inv_freq: torch.Tensor):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)


if __name__ == "__main__":
    rotary_dim = 128
    theta = 500000.0
    inv_freq = compute_inv_freq(rotary_dim, theta)
    print(inv_freq)

    new_freqs = apply_scaling(inv_freq)
    print(new_freqs)
