"""Shared colour helpers: PQ / sRGB transfer functions and BT.2020 constants."""

import torch

BT2020_LUMA = (0.2627, 0.6780, 0.0593)

# D65 linear BT.2020 -> linear BT.709. Out-of-gamut values are clipped by the
# caller; this is simple gamut clipping, not gamut compression.
RGB2020_TO_709 = (
    (1.660491, -0.587641, -0.072850),
    (-0.124550, 1.132900, -0.008349),
    (-0.018151, -0.100579, 1.118730),
)


def pq_decode(code):
    p = code.clip(0, 1) ** (1 / 78.84375)
    return 10000 * ((p - 0.8359375).clip(min=0) / (18.8515625 - 18.6875 * p)) ** (
        1 / 0.1593017578125
    )


def pq_encode(nits):
    p = (nits.clip(0, 10000) / 10000) ** 0.1593017578125
    return ((0.8359375 + 18.8515625 * p) / (1 + 18.6875 * p)) ** 78.84375


def srgb_encode(linear):
    x = linear.clip(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
