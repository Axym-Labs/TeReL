"""Locally connected (no weight sharing) layers as masked dense layers.

Each output neuron keeps its own weights over a spatial receptive field, so the
TeReL update of a neuron still uses only its own presynaptic inputs.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def receptive_field_mask(input_shape, channels, kernel, stride, padding):
    """Boolean (out_features, in_features) mask for channel-major flattened maps."""
    in_channels, height, width = input_shape
    out_height = (height + 2 * padding - kernel) // stride + 1
    out_width = (width + 2 * padding - kernel) // stride + 1
    spatial = torch.zeros(out_height * out_width, height * width, dtype=torch.bool)
    for oy in range(out_height):
        for ox in range(out_width):
            for ky in range(kernel):
                for kx in range(kernel):
                    iy, ix = oy * stride - padding + ky, ox * stride - padding + kx
                    if 0 <= iy < height and 0 <= ix < width:
                        spatial[oy * out_width + ox, iy * width + ix] = True
    # out index = c_out * (H'W') + position; in index = c_in * (HW) + pixel
    mask = spatial.repeat(channels, in_channels)
    return mask, (channels, out_height, out_width)


class MaskedLinear(nn.Linear):
    def __init__(self, mask: torch.Tensor):
        super().__init__(mask.shape[1], mask.shape[0])
        self.register_buffer("mask", mask.to(torch.float32))
        fan_in = int(mask[0].sum())
        bound = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound).mul_(self.mask)
            self.bias.uniform_(-bound, bound)

    def forward(self, values):
        return F.linear(values, self.weight * self.mask, self.bias)


def localize(layers: nn.ModuleList, spec, input_shape):
    """Replace the dense layers of `layers` in place according to `spec`."""
    if len(spec) != len(layers):
        raise ValueError("locally_connected needs one (channels, kernel, stride, padding) per layer")
    shape = tuple(input_shape)
    for index, (channels, kernel, stride, padding) in enumerate(spec):
        mask, shape = receptive_field_mask(shape, channels, kernel, stride, padding)
        dense = layers[index]
        if tuple(mask.shape) != (dense.out_features, dense.in_features):
            raise ValueError(f"layer {index}: hidden_dims do not match locally connected shape {shape}")
        layers[index] = MaskedLinear(mask).to(dense.weight.device)
    return layers
