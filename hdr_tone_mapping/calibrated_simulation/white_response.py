"""Scale measured white response, correcting only shadows toward identity."""
import csv
import numpy as np
from scipy.interpolate import PchipInterpolator


def monotone_regression(values):
    blocks=[]
    for value in values:
        blocks.append([float(value),1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            right,left=blocks.pop(),blocks.pop()
            count=left[1]+right[1]
            blocks.append([(left[0]*left[1]+right[0]*right[1])/count,count])
    return np.concatenate([np.full(count,mean) for mean,count in blocks])


def aces_filmic(x):
    x=np.maximum(x,0)
    return np.clip(x*(2.51*x+.03)/(x*(2.43*x+.59)+.14),0,1)


class WhiteResponse:
    def __init__(self,path,source_peak=10000,exposure_nits=100,shadow_start=.01,shadow_end=.1):
        with open(path,newline='',encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream,skipinitialspace=True))
        self.pixel=np.array([float(row['pixel']) for row in rows])
        self.measured=np.array([float(row['Y']) for row in rows])
        if (len(rows)<3 or not np.all(np.isfinite(self.pixel)) or
            not np.all(np.isfinite(self.measured)) or np.any(self.pixel<=0) or
            np.any(np.diff(self.pixel)<=0) or np.any(self.measured<=0)):
            raise ValueError('Expected positive finite Y and increasing positive pixel')
        if (not np.all(np.isfinite([source_peak,exposure_nits,shadow_start,shadow_end]))
            or source_peak<1000 or exposure_nits<=0 or not 0<shadow_start<shadow_end<1):
            raise ValueError('Invalid plotting range, exposure or shadow transition')
        self.source_peak,self.exposure=source_peak,exposure_nits
        self.shadow_start,self.shadow_end=shadow_start,shadow_end
        self.monotone=monotone_regression(self.measured)
        self.peak=float(self.monotone[-1])
        self.response=PchipInterpolator(np.log(self.pixel),np.log(self.monotone),extrapolate=False)
        mask=(self.monotone>=.01*self.peak)&(self.monotone<=.2*self.peak)
        if np.count_nonzero(mask)<2:
            raise ValueError('Not enough mid-range measurements')
        self.gain=float(np.exp(np.mean(np.log(self.monotone[mask]/self.pixel[mask]))))
        # A monotone Hermite bridge guarantees no local reversal in the blend.
        # Below start: identity. Above end: exactly the previous measured curve.
        self.bridges={}
        from scipy.interpolate import CubicHermiteSpline
        for peak in (400,1000):
            lo,hi=peak*shadow_start,peak*shadow_end
            yhi=float(self.baseline(hi,peak))
            scale=peak/self.peak*self.gain
            drive=hi/scale
            endpoint_slope=float(self.response.derivative()(np.log(drive)))
            secant=np.log(yhi/lo)/np.log(hi/lo)
            if secant<=0 or endpoint_slope<0 or 1+endpoint_slope>3*secant:
                raise ValueError('Shadow interval cannot support a monotone C1 bridge; widen it')
            self.bridges[peak]=CubicHermiteSpline(np.log([lo,hi]),np.log([lo,yhi]),
                                                [1,endpoint_slope],extrapolate=False)

    def forward(self,pixel):
        x=np.maximum(np.asarray(pixel,dtype=float),0)
        y=np.exp(self.response(np.log(np.clip(x,self.pixel[0],self.pixel[-1]))))
        return np.where(x<self.pixel[0],x*self.monotone[0]/self.pixel[0],y)

    def baseline(self,nits,peak):
        a=peak/self.peak
        return np.minimum(a*self.forward(np.maximum(nits,0)/(a*self.gain)),peak)

    def spline(self,nits,peak):
        x=np.maximum(np.asarray(nits,dtype=float),0)
        lo,hi=peak*self.shadow_start,peak*self.shadow_end
        bridge=np.exp(self.bridges[peak](np.log(np.clip(x,lo,hi))))
        return np.where(x<=lo,x,np.where(x<hi,bridge,self.baseline(x,peak)))

    def aces(self,nits):
        return 100*aces_filmic(np.asarray(nits)/self.exposure)

    def summary(self):
        return dict(model='Scaled measured white response with shadow-only identity correction',
                    measured_peak_nits=self.peak,target_peaks_nits=[400,1000],midrange_gain=self.gain,
                    baseline='T_P(L)=(P/P_measured)*M(L/((P/P_measured)*gain))',
                    gain_fit='Geometric mean of Y/pixel at 1%-20% measured peak',
                    shadow_start_fraction=self.shadow_start,shadow_end_fraction=self.shadow_end,
                    shadow_ranges_nits={str(p):[p*self.shadow_start,p*self.shadow_end] for p in (400,1000)},
                    correction='Identity below start; log-log monotone C1 cubic bridge; original measured-shape curve unchanged above end',
                    shadow_parameter_status='Explicit initial modeling choices, not inferred from CSV',
                    input_units='CSV pixel used as nominal input luminance coordinate for scaled simulation; verify acquisition units',
                    aces_exposure_nits=self.exposure,sdr_white_nits=100,
                    aces_exposure_status='Initial assumption; independent of CSV',
                    limitation='Simulation based on measured shape, not inverse display calibration',
                    isotonic_max_adjustment_nits=float(np.max(np.abs(self.monotone-self.measured))))


def scale_luminance(rgb,target,weights):
    y=rgb@weights
    gain=np.divide(target,y,out=np.zeros_like(y),where=y>0)
    return rgb*gain[...,None]
