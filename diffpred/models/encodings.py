import numpy as np
import torch


# Fourier Feature Embedding
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        # Handle both single inputs and batched inputs
        original_shape = x.shape

        # Reshape to (total_elements, )
        x_flat = x.reshape(-1)

        # Apply Fourier mapping to the flattened input
        x_mapped = x_flat.ger((2 * np.pi * self.freqs).to(x.dtype))
        x_mapped = torch.cat([x_mapped.cos(), x_mapped.sin()], dim=1)

        # Reshape back to include batch dimensions
        output_shape = original_shape + (x_mapped.shape[-1],)
        return x_mapped.reshape(output_shape)
