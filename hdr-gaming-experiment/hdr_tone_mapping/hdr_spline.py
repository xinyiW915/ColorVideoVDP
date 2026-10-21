"""Calibrated HDR spline: knee and highlight roll-off taken from the measured white response.

WhiteResponse fits the curve from the CSV (NumPy, used for plots and parameters);
HDRSpline applies it to video frames (PyTorch, CPU or CUDA).
"""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from hdr_tone_mapping.color_encoding import pq_encode

TARGET_PEAKS = (400, 1000)
WHITE_CSV = Path(__file__).resolve().parent / "asushdr_display" / "resp_asushdr_pq1000_white.csv"


def monotone_regression(values):
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    return np.concatenate([np.full(count, mean) for mean, count in blocks])


class WhiteResponse:
    def __init__(self, path):
        with open(path, newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        self.pixel = np.array([float(row["pixel"]) for row in rows])
        self.measured = np.array([float(row["Y"]) for row in rows])
        if (
            len(rows) < 3
            or not np.all(np.isfinite(self.pixel))
            or not np.all(np.isfinite(self.measured))
            or np.any(self.pixel <= 0)
            or np.any(np.diff(self.pixel) <= 0)
            or np.any(self.measured <= 0)
        ):
            raise ValueError("Expected positive finite Y and increasing positive pixel")
        self.monotone = monotone_regression(self.measured)
        self.peak = float(self.monotone[-1])
        self.response = PchipInterpolator(
            np.log(self.pixel), np.log(self.monotone), extrapolate=False
        )
        mask = (self.monotone >= 0.01 * self.peak) & (self.monotone <= 0.2 * self.peak)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Not enough mid-range measurements")
        # Mid-range Y/pixel, used only to place the simulation curve for plotting.
        self.midrange_gain = float(
            np.exp(np.mean(np.log(self.monotone[mask] / self.pixel[mask])))
        )
        # Plateau: first sample within 0.5% of the measured peak; later samples
        # only differ by measurement noise.
        self.plateau_pixel = float(
            self.pixel[np.argmax(self.monotone >= 0.995 * self.peak)]
        )
        self.plateau = float(self.forward(self.plateau_pixel))
        # Knee: last point below the plateau where the measured log-log slope
        # falls through 1. Above it the display only compresses, with slope 1
        # at the knee, so identity below joins the measured roll-off C1.
        u = np.linspace(np.log(self.pixel[0]), np.log(self.plateau_pixel), 400001)
        slope = self.response.derivative()(u)
        cross = np.flatnonzero((slope[:-1] >= 1) & (slope[1:] < 1))
        if cross.size == 0:
            raise ValueError("Measured response never falls below slope 1")
        self.knee_pixel = float(np.exp(u[cross[-1]]))
        self.knee_response = float(self.forward(self.knee_pixel))
        self.knee = self.knee_response / self.plateau
        if not 0 < self.knee < 1 or np.any(slope[cross[-1] + 1 :] > 1):
            raise ValueError("Measured roll-off is not compressive above the knee")

    def forward(self, pixel):
        """Measured display response M(pixel) in nits (monotone PCHIP in log-log)."""
        x = np.maximum(np.asarray(pixel, dtype=float), 0)
        y = np.exp(self.response(np.log(np.clip(x, self.pixel[0], self.pixel[-1]))))
        return np.where(x < self.pixel[0], x * self.monotone[0] / self.pixel[0], y)

    def simulation(self, nits, peak):
        # Measured shape scaled to the target peak, mid-range aligned to 1:1;
        # shown for reference, highlights may exceed 1:1.
        a = peak / self.peak
        drive = np.maximum(nits, 0) / (a * self.midrange_gain)
        return np.minimum(a * self.forward(drive), peak)

    def spline(self, nits, peak):
        # Identity up to knee K = knee * peak; above it, the measured roll-off
        # scaled so its knee lands on (K, K) and its plateau on the target peak.
        # HDRSpline evaluates the same curve in PyTorch.
        x = np.clip(np.asarray(nits, dtype=float), 0, 10000)
        knee = self.knee * peak
        rolloff = knee * self.forward(self.knee_pixel * x / knee) / self.knee_response
        return np.where(x <= knee, x, np.minimum(rolloff, peak))

    def summary(self):
        return {
            "model": "Identity to measured knee, then measured roll-off scaled to target peak",
            "measured_peak_nits": self.peak,
            "measured_plateau_nits": self.plateau,
            "plateau_pixel": self.plateau_pixel,
            "knee_pixel": self.knee_pixel,
            "knee_fraction_of_peak": self.knee,
            "knee_nits": {str(p): p * self.knee for p in TARGET_PEAKS},
            "peak_reached_at_input_nits": {
                str(p): p * self.knee * self.plateau_pixel / self.knee_pixel
                for p in TARGET_PEAKS
            },
            "knee_fit": "Last point below plateau where measured log-log slope falls through 1",
            "curve": "T(L)=L for L<=K; K*M(x_k*L/K)/M(x_k) above, capped at P; K=knee*P",
            "channels": "Same curve applied to each linear BT.2020 RGB channel",
            "midrange_gain": self.midrange_gain,
            "simulation": "Plot only: (P/P_measured)*M(L/((P/P_measured)*midrange_gain))",
            "limitation": "Knee and roll-off follow the measured display; not inverse display calibration",
            "isotonic_max_adjustment_nits": float(
                np.max(np.abs(self.monotone - self.measured))
            ),
        }


class HDRSpline:
    """Calibrated HDR spline for one target peak: linear BT.2020 RGB nits -> PQ."""

    def __init__(self, model, peak, device="cpu"):
        self.device = torch.device(device)
        self.model, self.peak = model, peak
        self.knee = model.knee * peak
        # PCHIP pieces of the measured response, evaluated exactly (no LUT).
        self.knots = torch.tensor(
            model.response.x, dtype=torch.float32, device=self.device
        )
        self.coefficients = torch.tensor(
            model.response.c, dtype=torch.float32, device=self.device
        )

    def response(self, log_pixel):
        """log M(pixel); same cubic coefficients and Horner form as SciPy PPoly."""
        index = (
            torch.bucketize(log_pixel.contiguous(), self.knots, right=True)
            .sub_(1)
            .clamp_(0, self.knots.numel() - 2)
        )
        d = log_pixel - self.knots[index]
        c = self.coefficients
        return ((c[0, index] * d + c[1, index]) * d + c[2, index]) * d + c[3, index]

    def __call__(self, rgb):
        """Same curve as WhiteResponse.spline, applied to each RGB channel."""
        model, knee = self.model, self.knee
        x = rgb.clamp(0, 10000)
        # Only drives above the knee pixel are used (x > knee).
        drive = (x * (model.knee_pixel / knee)).clamp(
            model.knee_pixel, float(model.pixel[-1])
        )
        rolloff = (self.response(drive.log()).exp() * (knee / model.knee_response)).clamp(
            max=self.peak
        )
        # Every channel stays <= peak; saturated highlights move towards white.
        return pq_encode(torch.where(x <= knee, x, rolloff))

    def summary(self):
        return {
            "mapping": "Identity to measured knee, measured roll-off to target peak; per RGB channel",
            "peak_nits": self.peak,
            "knee_nits": self.knee,
            "output": "BT.2020 / PQ",
        }
