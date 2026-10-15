"""CUDA evaluation of the simple spline and shared luminance-only ACES."""
import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline
from hdr_tone_mapping.sdr_mapping import encode_tensor


class ToneMapper:
    def __init__(self, decode_lut, matrix, source_peak=10000, knee_ratio=.5, exposure_nits=100):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu')
        if not (0 < knee_ratio <= .5 and 1000 < source_peak <= 10000 and exposure_nits > 0):
            raise ValueError('Invalid spline parameters or exposure')
        self.source_peak, self.knee_ratio, self.exposure = source_peak, knee_ratio, exposure_nits
        self.lut = torch.tensor(decode_lut, dtype=torch.float32, device='cuda')
        self.weights = torch.tensor([.2627,.6780,.0593], device='cuda')
        self.matrix = torch.tensor(matrix, dtype=torch.float32, device='cuda')
        self.curves = {}
        for peak in (400,1000):
            a,b,c = np.log([peak*knee_ratio,peak,source_peak])
            delta = b-a
            slope = min(.2,1.2*delta/(c-b))
            curve = CubicHermiteSpline([a,b,c],[a,a+.6*delta,b],[1,slope,0],extrapolate=False)
            self.curves[peak] = (torch.tensor(curve.x,dtype=torch.float32,device='cuda'),
                                 torch.tensor(curve.c,dtype=torch.float32,device='cuda'))

    def spline(self,y,peak):
        knee = peak*self.knee_ratio
        values = y.clamp(0,self.source_peak)
        x = values.clamp(knee,self.source_peak).log()
        knots,coeff = self.curves[peak]
        index = (torch.bucketize(x.contiguous(),knots,right=True)-1).clamp(0,len(knots)-2)
        d = x-knots[index]
        shoulder = (((coeff[0,index]*d+coeff[1,index])*d+coeff[2,index])*d+coeff[3,index]).exp()
        return torch.where(values<=knee,values,shoulder)

    @staticmethod
    def pq_encode(nits):
        p = (nits.clamp(0,10000)/10000).pow(.1593017578125)
        return ((.8359375+18.8515625*p)/(1+18.6875*p)).pow(78.84375)

    @torch.inference_mode()
    def prepare(self,codes):
        index = torch.from_numpy(codes.astype(np.int32)).to('cuda').long()
        rgb = self.lut[index]
        return rgb,(rgb*self.weights).sum(dim=-1)

    @torch.inference_mode()
    def encode(self,rgb,y,mode):
        if mode.startswith('sdr'):
            encoded, _ = encode_tensor(rgb, y, self.matrix, self.exposure)
        else:
            target = self.spline(y,1000 if '1000' in mode else 400)
            gain = torch.where(y>0,target/y.clamp_min(torch.finfo(y.dtype).tiny),1)
            p = rgb*gain[...,None]
            encoded = self.pq_encode(p)
        return encoded.cpu().numpy()
