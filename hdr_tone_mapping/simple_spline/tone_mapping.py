"""Luminance-only HDR splines and SDR ACES Filmic."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline


def pq_decode(value):
    p = np.clip(value, 0, 1) ** (1 / 78.84375)
    return 10000 * (np.maximum(p - 0.8359375, 0) /
                    (18.8515625 - 18.6875 * p)) ** (1 / 0.1593017578125)


def pq_encode(nits):
    p = (np.clip(nits, 0, 10000) / 10000) ** 0.1593017578125
    return ((0.8359375 + 18.8515625 * p) / (1 + 18.6875 * p)) ** 78.84375


def hdr_spline(nits, peak, source_peak=10000, knee_ratio=0.5):
    """Identity below knee; C1 monotone cubic spline in log(input), log(output).

    The two shoulder pieces join at input=peak. The final endpoint is
    (source_peak, peak), with zero slope; this is a content transform,
    not a fit or inverse calibration of the measured ASUS display.
    """
    if not (0 < knee_ratio <= 0.5 and 0 < peak < source_peak <= 10000):
        raise ValueError("Require 0 < knee_ratio <= .5 and 0 < peak < source_peak <= 10000")
    knee = peak * knee_ratio
    a, b, c = np.log([knee, peak, source_peak])
    delta = b - a
    middle_slope = min(0.2, 1.2 * delta / (c - b))
    spline = CubicHermiteSpline([a, b, c], [a, a + 0.6 * delta, b],
                               [1, middle_slope, 0], extrapolate=False)
    values = np.clip(np.asarray(nits, dtype=np.float64), 0, source_peak)
    shoulder = np.exp(spline(np.log(np.clip(values, knee, source_peak))))
    return np.where(values <= knee, values, shoulder)


BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593])


def map_hdr_luminance(rgb_nits, peak, source_peak=10000, knee_ratio=0.5):
    """Map Y with the spline and preserve linear RGB ratios, including black.

    Input is nonnegative linear BT.2020 RGB in absolute nits. The peak
    constrains luminance Y, not each RGB component. Since the curve only
    compresses, valid PQ input components remain within the PQ range.
    """
    rgb = np.asarray(rgb_nits)
    luminance = rgb @ BT2020_LUMA
    mapped = hdr_spline(luminance, peak, source_peak, knee_ratio)
    gain = np.divide(mapped, luminance, out=np.ones_like(mapped), where=luminance > 0)
    return rgb * gain[..., np.newaxis]


def sdr_aces_fit(rgb_nits, exposure_nits=100):
    """Narkowicz ACES Filmic, linear RGB input/output, not the full ACES pipeline.

    Source: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    exposure_nits defines the absolute input mapped to x=1 (output ~0.804).
    """
    if exposure_nits <= 0:
        raise ValueError("exposure_nits must be positive")
    x = np.maximum(rgb_nits, 0) / exposure_nits
    return np.clip((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14), 0, 1)


def srgb_encode(linear):
    x = np.clip(linear, 0, 1)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)


# D65 linear BT.2020 -> linear BT.709. Negative/out-of-gamut values are clipped
# before SDR ACES Filmic; this is simple gamut clipping, not gamut compression.
RGB2020_TO_709 = np.array([[1.660491, -.587641, -.072850],
                         [-.124550, 1.132900, -.008349],
                         [-.018151, -.100579, 1.118730]], dtype=np.float32)
