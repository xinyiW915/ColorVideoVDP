"""SDR ACES Filmic (Narkowicz fit) applied per RGB channel, as in the original post.

Source: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
Linear BT.709 input is multiplied by exposure, the curve is applied to each RGB
component, and gamma (sRGB) encoding follows. Output 1.0 is the 100-nit SDR white.
"""

import torch

from hdr_tone_mapping.color_encoding import RGB2020_TO_709, srgb_encode

SDR_WHITE_NITS = 100
# A pixel counts as out of the BT.709 gamut only if a component is below
# -1% of the exposure (-1 nit at the default 100). Smaller negatives come
# mostly from noise in near-black pixels; they are still clamped to 0.
OUT_OF_GAMUT_THRESHOLD = 0.01


def aces_filmic(x):
    """ACESFilm() from the post on relative exposure; NumPy array or tensor.

    Negative inputs (out-of-gamut BT.709 components) are clamped to 0 first,
    since the rational curve is not monotone below 0.
    """
    x = x.clip(min=0)
    return (x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)).clip(0, 1)


class SDRAces:
    """Linear BT.2020 RGB nits -> sRGB-encoded BT.709; counts out-of-gamut pixels."""

    def __init__(self, device="cpu", exposure_nits=100):
        if exposure_nits <= 0:
            raise ValueError("exposure_nits must be positive")
        self.device = torch.device(device)
        self.exposure = exposure_nits
        self.to_709 = torch.tensor(RGB2020_TO_709, device=self.device)
        self.out_of_gamut_pixels = 0
        self.total_pixels = 0

    def curve(self, nits):
        """Output nits for input nits on one channel (or a neutral grey); NumPy array or tensor."""
        return SDR_WHITE_NITS * aces_filmic(nits / self.exposure)

    def __call__(self, rgb):
        # Exposure, then BT.2020 -> BT.709; both linear, so the order does not matter.
        relative = (rgb @ self.to_709.T) / self.exposure
        outside = (relative < -OUT_OF_GAMUT_THRESHOLD).any(dim=-1)
        self.out_of_gamut_pixels += int(outside.sum().item())
        self.total_pixels += outside.numel()
        return srgb_encode(aces_filmic(relative))

    def summary(self):
        return {
            "mapping": "Narkowicz ACES Filmic applied per linear BT.709 RGB channel",
            "exposure_nits": self.exposure,
            "exposure_status": "Initial assumption; independent of CSV",
            "sdr_white_nits": SDR_WHITE_NITS,
            "out_of_gamut_pixels": self.out_of_gamut_pixels,
            "out_of_gamut_percent": 100
            * self.out_of_gamut_pixels
            / max(self.total_pixels, 1),
            "out_of_gamut_threshold_nits": OUT_OF_GAMUT_THRESHOLD * self.exposure,
            "out_of_gamut_definition": "Any BT.709 component below -threshold; smaller negatives are mostly near-black noise",
            "out_of_gamut_handling": "All negative BT.709 components clamped to 0 before the curve",
            "output": "BT.709 / sRGB",
        }
