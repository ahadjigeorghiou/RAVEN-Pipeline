import numpy as np
from astropy.io import fits
import sys
import warnings
from Features import utils
warnings.filterwarnings('once')


def dopolyfit(win, d, ni, sigclip):
    base = np.polyfit(win[:,0],win[:,1],w=1.0/win[:,2],deg=d)
    # for n iterations, clip sigma, redo polyfit
    for iter in range(ni):
        #winsigma = np.std(win[:,1]-np.polyval(base,win[:,0]))
        offset = np.abs(win[:,1]-np.polyval(base,win[:,0]))/win[:,2]
        
        if (offset<sigclip).sum()>int(0.8*len(win[:,0])):
            clippedregion = win[offset<sigclip,:]
        else:
            clippedregion = win[offset<np.average(offset)]
            
        base = np.polyfit(clippedregion[:,0],clippedregion[:,1],w=1.0/np.power(clippedregion[:,2],2),deg=d)
    return base


def CheckForGaps(dat,centidx,winlowbound,winhighbound,gapthresh):
    diffshigh = np.diff(dat[centidx:winhighbound,0])
    gaplocshigh = np.where(diffshigh>gapthresh)[0]
    highgap = len(gaplocshigh)>0
    diffslow = np.diff(dat[winlowbound:centidx,0])
    gaplocslow = np.where(diffslow>gapthresh)[0]
    lowgap = len(gaplocslow)>0
    return lowgap, highgap, gaplocslow,gaplocshigh


def formwindow(datcut,dat,cent,size,boxsize,gapthresh,expectedpoints,cadence):
    winlowbound = np.searchsorted(datcut[:,0],cent-size/2.)
    winhighbound = np.searchsorted(datcut[:,0],cent+size/2.)
    boxlowbound = np.searchsorted(dat[:,0],cent-boxsize/2.)
    boxhighbound = np.searchsorted(dat[:,0],cent+boxsize/2.)
    centidx = np.searchsorted(datcut[:,0],cent)

    if centidx==boxlowbound:
        centidx += 1
    if winhighbound == len(datcut[:,0]):
        winhighbound -= 1
    flag = 0

    lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)

    if winlowbound == 0:
        lowgap = True
        gaplocslow = [-1]
    if winhighbound == len(datcut[:,0]):
         highgap = True
         gaplocshigh = [len(datcut[:,0]) -centidx]
    
    if highgap:
        if lowgap:
            winhighbound = centidx + gaplocshigh[0]
            winlowbound = winlowbound + 1 + gaplocslow[-1]
        else:
            winhighbound = centidx + gaplocshigh[0]
            winlowbound = np.searchsorted(datcut[:,0],datcut[winhighbound,0]-size)
            lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)
            if lowgap:
                winlowbound = winlowbound + 1 + gaplocslow[-1] #uses reduced fitting section
    else:
        if lowgap:
            winlowbound = winlowbound + 1 + gaplocslow[-1]            
            winhighbound =  np.searchsorted(datcut[:,0],datcut[winlowbound,0]+size)
            lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)
            if highgap:
                winhighbound = centidx + gaplocshigh[0] #uses reduced fitting section

    #window = np.concatenate((dat[winlowbound:boxlowbound,:],dat[boxhighbound:winhighbound,:]))
    window = datcut[winlowbound:winhighbound,:]
    if len(window[:,0]) < 20:
        flag = 1
    box = dat[boxlowbound:boxhighbound,:]

    return window,boxlowbound,boxhighbound,flag


def polyflatten(lc,winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,t0=0.,transitcut=False,tc_per=0,tc_t0=0,tc_tdur=0,divide=True):

    lcdetrend = np.zeros(len(lc[:,0]))
    trend = np.zeros(len(lc[:,0]))

    #general setup
    lenlc = lc[-1,0]
    nsteps = np.ceil(lenlc/stepsize).astype('int')
    stepcentres = np.arange(nsteps)/float(nsteps) * lenlc + stepsize/2.
    cadence = np.median(np.diff(lc[:,0]))
    
    expectedpoints = winsize/2./cadence

    if transitcut:
        if isinstance(tc_per, np.ndarray) or isinstance(tc_per, list):
            timecut, fluxcut = lc[:, 0].copy() + t0, lc[:, 1].copy()
            errcut = lc[:, 2].copy()
            for cutidx in range(len(tc_per)):
                timecut, fluxcut, errcut = CutTransits(timecut,fluxcut,errcut,
                										tc_t0[cutidx],
                										tc_per[cutidx],tc_tdur[cutidx])
        else:
            timecut, fluxcut, errcut = CutTransits(lc[:,0]+t0,lc[:,1],
            										lc[:,2],tc_t0,tc_per,tc_tdur)
        lc_tofit = np.zeros([len(timecut),3])
        lc_tofit[:,0] = timecut-t0
        lc_tofit[:,1] = fluxcut
        lc_tofit[:,2] = errcut
    else:
        lc_tofit = lc

    #for each step centre:
    for s in range(nsteps):
        stepcent = stepcentres[s]
        winregion,boxlowbound,boxhighbound,flag = formwindow(lc_tofit,lc,stepcent,winsize,stepsize,gapthreshold,expectedpoints,cadence)  #should return window around box not including box

        if not flag:
            baseline = dopolyfit(winregion,polydegree,niter,sigmaclip)
            if divide:
                lcdetrend[boxlowbound:boxhighbound] = lc[boxlowbound:boxhighbound,1] / np.polyval(baseline,lc[boxlowbound:boxhighbound,0])
            else:
                lcdetrend[boxlowbound:boxhighbound] = lc[boxlowbound:boxhighbound,1] - np.polyval(baseline,lc[boxlowbound:boxhighbound,0])
            trend[boxlowbound:boxhighbound] = np.polyval(baseline,lc[boxlowbound:boxhighbound,0])
        else:
            if divide:
                lcdetrend[boxlowbound:boxhighbound] = np.ones(boxhighbound-boxlowbound)
            else:
                lcdetrend[boxlowbound:boxhighbound] = np.zeros(boxhighbound-boxlowbound)
            trend[boxlowbound:boxhighbound] = np.nan
    
    output = np.zeros_like(lc)
    output[:,0] = lc[:,0] + t0
    output[:,1] = lcdetrend
    output[:,2] = lc[:,2]
    return output, trend


def CutTransits(time,flux,err,t0,per,tdur):
    if per == 0 or np.isnan(per):
        per1 = time[-1] - (t0-tdur*0.5)
        per2 = t0-tdur*0.5 - (time[0])
        per = np.max((per1, per2))
    phase = utils.phasefold(time, per, t0 - per*0.5) - 0.5
    tdur_p = tdur/per
    if tdur_p > 0.2:
        #print('Transit duration greater than 20% of the phase, transits not removed.')
        return time, flux, err
    intransit = np.abs(phase)<=0.5*tdur_p
    return time[~intransit], flux[~intransit], err[~intransit]


def TESSflatten(lcurve, split=True, winsize=2.5, 
				stepsize=0.15,polydeg=3,niter=10,sigmaclip=4.,gapthresh=100.,
				transitcut=False,tc_per=0,tc_t0=0,tc_tdur=0,divide=True, return_trend=False, centroid=False):
    """
    lcurve - lightcurve, ndarray with first column time, second flux, third error (used for weighting the polynomial fit).
    split - whether to split the lightcurve on each TESS orbit. Assumes start of lightcurve is start of a sector.
    winsize - size of region to fit polynomial to
    stepsize - size of region that that polynomial is used to detrend
    polydeg - degree of polynomial
    niter - number of iterations for each window region. Each iteration, the polynomial is fit, outliers are ignored, and the fit is repeated.
    sigmaclip - clip threshold when iterating
    gapthresh - the code will avoid fitting polynomials over gaps larger than this.
    sectorstart - the time of start of the first sector in the data (needed for identifying orbit splits)
    transitcut - mask transits. Should be an integer showing how many planets to mask
    tc_per etc - parameters of transit to mask. If transitcut >1, these need to be indexable, e.g. a list of the periods
    divide - normalise by the fitted baseline if true (e.g. for flux timeseries), otherwise subtract the baseline
    """
    time_diff = np.diff(lcurve[:, 0])
    cadence = np.median(time_diff)
    flatlc = np.zeros(1)
    trend_lst = []

    if split:

        gap_indices = np.argwhere(time_diff >= 1.0)
        gap_indices += 1
        gap_indices = np.append(gap_indices, len(lcurve[:, 0]))

        expectedpoints = np.ceil(winsize / cadence)
            
        i = 0
        falsegap = []
        for idx in gap_indices:
            if len(lcurve[:,0][i:idx]) < expectedpoints:
                if i == 0:
                    falsegap.append(idx)
                else:
                    falsegap.append(i)
                    i = idx
            else:
                i = idx

        gap_indices = [x for x in gap_indices if x not in falsegap]
        
        if centroid:
            discontinuities = np.where(np.abs(np.diff(lcurve[:,1])) > 1e-2)[0]
            discontinuities += 1
            gap_indices = list(gap_indices) + [x for x in discontinuities if x not in gap_indices]
            gap_indices = sorted(gap_indices)
                
        start = 0
        for end in gap_indices:
            lcseg = lcurve[start:end, :]
            if 0 in lcseg.shape:
                continue
            t0 = lcseg[0, 0].copy()
            lcseg[:, 0] -= lcseg[0, 0]
            lcseg_flat, trend = polyflatten(lcseg, winsize, stepsize, polydeg, niter,
                                     sigmaclip, gapthresh, t0=t0,
                                     transitcut=transitcut, tc_per=tc_per,
                                     tc_t0=tc_t0, tc_tdur=tc_tdur, divide=divide)
            lcseg_flat = lcseg_flat[:, 1]
            lcseg[:, 0] += t0
            flatlc = np.hstack((flatlc, lcseg_flat))
            trend_lst.append(trend)
            start = end
    flatlc = np.array(flatlc)
    trend = np.concatenate(trend_lst)
    
    if return_trend:
        return flatlc[1:], trend
    else:
        return flatlc[1:]

