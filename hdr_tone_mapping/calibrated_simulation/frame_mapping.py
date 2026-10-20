"""CUDA evaluation of the existing measured splines (no LUT approximation)."""

import numpy as np
import torch

from hdr_tone_mapping.sdr_mapping import encode_tensor


class ToneMapper:
    def __init__(self, model, decode_lut, rgb2020_to_709, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA tone mapping requires a CUDA-enabled PyTorch and GPU"
            )
        self.model = model
        self.decode_lut = torch.as_tensor(decode_lut, device=self.device)
        self.weights = torch.tensor([0.2627, 0.6780, 0.0593], device=self.device)
        self.matrix = torch.as_tensor(rgb2020_to_709, device=self.device)
        self.response = self._upload(model.response)
        self.bridges = {
            peak: self._upload(spline) for peak, spline in model.bridges.items()
        }

    def _upload(self, spline):
        return (
            torch.tensor(spline.x, dtype=torch.float32, device=self.device),
            torch.tensor(spline.c, dtype=torch.float32, device=self.device),
        )

    @staticmethod
    def _evaluate(x, spline):
        knots, coefficients = spline
        index = (
            torch.bucketize(x.contiguous(), knots, right=True)
            .sub_(1)
            .clamp_(0, knots.numel() - 2)
        )
        delta = x - knots[index]
        # Same cubic coefficients and Horner evaluation as SciPy PPoly.
        return (
            (coefficients[0, index] * delta + coefficients[1, index]) * delta
            + coefficients[2, index]
        ) * delta + coefficients[3, index]

    def spline(self, y, peak):
        model = self.model
        lo, hi = peak * model.shadow_start, peak * model.shadow_end
        bridge = self._evaluate(y.clamp(lo, hi).log(), self.bridges[peak]).exp()
        a = peak / model.peak
        drive = y / (a * model.gain)
        response = self._evaluate(
            drive.clamp(float(model.pixel[0]), float(model.pixel[-1])).log(),
            self.response,
        ).exp()
        response = torch.where(
            drive < model.pixel[0],
            drive * (model.monotone[0] / model.pixel[0]),
            response,
        )
        baseline = (a * response).clamp(max=peak)
        return torch.where(y <= lo, y, torch.where(y < hi, bridge, baseline))

    @staticmethod
    def pq_encode(nits):
        p = (nits.clamp(0, 10000) / 10000).pow(0.1593017578125)
        return ((0.8359375 + 18.8515625 * p) / (1 + 18.6875 * p)).pow(78.84375)

    @staticmethod
    def srgb_encode(linear):
        x = linear.clamp(0, 1)
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - 0.055)

    @torch.inference_mode()
    def prepare(self, codes):
        # uint16 indexing is converted before transfer for broad Torch support.
        index = torch.from_numpy(codes.astype(np.int32)).to(self.device).long()
        rgb = self.decode_lut[index]
        y = (rgb * self.weights).sum(dim=-1)
        return rgb, y

    @torch.inference_mode()
    def encode(self, rgb, y, mode):
        is_sdr = mode.startswith("sdr")
        if is_sdr:
            encoded, clipped = encode_tensor(rgb, y, self.matrix, self.model.exposure)
            return encoded.cpu().numpy(), clipped
        target = self.spline(y, 1000 if "1000" in mode else 400)
        gain = torch.where(y > 0, target / y.clamp_min(torch.finfo(y.dtype).tiny), 0)
        mapped = rgb * gain[..., None]
        if mapped.max().item() > 10000 + 1e-3:
            raise ValueError("Mapped RGB exceeds PQ range")
        encoded = self.pq_encode(mapped)
        return encoded.cpu().numpy(), 0
