# This file is part of the TransitSOM code accompanying the paper Armstrong et al 2016
# Copyright (C) 2016 David Armstrong
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#utils functions file
import numpy as np
from astropy.io import fits
from CandidateSet import TESSselfflatten as tsf
from Features import utils as futils

def phasefold(time,per,t0):
    return np.mod(time-t0,per)/per

def WeightMeanClipped(vals):
    values = vals[:,0]
    weights = vals[:,1]
    mean = np.sum(values*weights)/(np.sum(weights))
    outliers = np.abs((values - mean)/vals[:,2]) > vals[0,3]
    mean = np.sum(values[~outliers]*weights[~outliers])/(np.sum(weights[~outliers]))
    return mean

def WeightMean(vals):
    values = vals[:,0]
    weights = vals[:,1]
    mean = np.sum(values*weights)/(np.sum(weights))
    return mean

def WeightMean_witherror(vals):
    values = vals[:,0]
    weights = vals[:,1]
    mean = np.sum(values*weights)/(np.sum(weights))
    error = np.sqrt(1./np.sum(weights))
    return mean,error

def GetBinnedVals(time,x,lowerr,higherr,bins,bin_edges,clip_outliers=0):

    ###
    #bins x using the mean, weighted by the inverse error squared
    #if clip_outliers is not zero, it will be used as a significance threshold. 
    #data more than clip_outliers*its error from the unclipped bin mean will be 
    #ignored in producing the final bin values
    ###
    
    tobin = np.zeros([len(x),4])
    tobin[:,0] = x
    tobin[:,1] = np.power(2./(higherr+lowerr),2)
    tobin[:,2] = (higherr+lowerr)/2.
    tobin[:,3] = np.ones(len(x))*clip_outliers
    
    binnumber = np.digitize(time, bin_edges)
    timemeans = np.zeros(bins)
    
    if clip_outliers:
        for i in range(bins):
            segment = tobin[binnumber == i+1]
            if segment.shape[0] > 0:
                timemeans[i] = WeightMeanClipped(segment)
            else:
                timemeans[i] = np.nan
    
    else:
        for i in range(bins):
            segment = tobin[binnumber == i+1]
            if segment.shape[0] > 0:
                timemeans[i] = WeightMean(segment)
            else:
                timemeans[i] = np.nan
    binerrors = np.zeros(len(timemeans))

    for bin in range(len(timemeans)):

        weights = tobin[binnumber==bin+1,1]
        values = tobin[binnumber==bin+1,0]
        if clip_outliers:
            outliers = np.abs((values - timemeans[bin])/tobin[binnumber==bin+1,2]) > clip_outliers
            binerrors[bin] = np.sqrt(1./np.sum(weights[~outliers])) 
        else:
            binerrors[bin] = np.sqrt(1./np.sum(weights))

    binnedtimes = (bin_edges[1:]+bin_edges[:-1])/2.

    return binnedtimes,timemeans,binerrors

def GetBinnedVals_nooutliers(time,x,lowerr,higherr,bins):
    tobin = np.zeros([len(x),3])
    tobin[:,0] = x
    tobin[:,1] = np.power(2./(higherr+lowerr),2)  #inverse errors squared
    tobin[:,2] = (higherr+lowerr)/2.

    bin_edges = np.linspace(time[0], time[-1], bins+1)
    bin_edges[-1]+=0.0001
    binnumber = np.digitize(time, bin_edges)
    timemeans = np.zeros(bins)
    binerrors = np.zeros(len(timemeans))
    for i in range(bins):
        segment = tobin[binnumber == i+1]
        if segment.shape[0] > 0:
            timemeans[i],binerrors[i] = WeightMean_witherror(segment)
        else:
            timemeans[i] = np.nan
            binerrors[i] = np.nan
    
    binnedtimes = (bin_edges[1:]+bin_edges[:-1])/2.
    return binnedtimes,timemeans,binerrors

def PrepareArrays(SOMarray,errorarray):

    #replace nans with interpolation, normalise transits
    
    for idx in range(SOMarray.shape[0]):
        mask = np.isnan(SOMarray[idx,:])
        #edge cases given closest valid bin value
        if np.sum(mask)<len(SOMarray[idx,mask])*0.8:  #need at least 20% of points
            SOMarray[idx,mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), SOMarray[idx,~mask],right=1.,left=1.)
            SOMarray[idx,:],errorarray[idx,:] = SOMNormalise(SOMarray[idx,:],errorarray[idx,:])
        else:
            SOMarray[idx,:] = np.ones(len(SOMarray[idx,:]))
            errorarray[idx,:] = np.ones(len(errorarray[idx,:]))
    return SOMarray,errorarray
    
def PrepareArray(SOMarray_single,errorarray_single):

    #replace nans with interpolation, normalise transits
    
    mask = np.isnan(SOMarray_single)
    #edge cases given closest valid bin value
    if np.sum(mask)<len(SOMarray_single)*0.8:  #need at least 20% of points
        SOMarray_single[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), SOMarray_single[~mask],right=1.,left=1.)
        SOMarray_single,errorarray_single = SOMNormalise(SOMarray_single,errorarray_single)
    else:
        SOMarray_single = np.ones(len(SOMarray_single))
        errorarray_single = np.ones(len(errorarray_single))
    return SOMarray_single,errorarray_single
    
def SOMNormalise(flux,errors):

    #normalises flux so that mean of lowest quarter of points is 0, and mean of highest quarter of points is 1.
    #assumes flux consists of 3 transit durations or similar, otherwise normalisation will be wrong.
    
    npoints = len(flux)
    sorted_flux = np.sort(flux)
    lowlevel = np.mean(sorted_flux[:int(npoints/4-1)])
    norm = 1./(np.mean(sorted_flux[-int(npoints/4):])-lowlevel)
    flux -= lowlevel
    flux *= norm
    errors *= norm
    errors[~np.isfinite(errors)] = 1.0  #catches infs in the error array
    return flux,errors


def load_injected(filename, flatten=True, winsize=2.0, t0=None, per=None, tdur=None):
    hdu = fits.open(filename)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['FLUX']

    nancut = np.isnan(time) | np.isnan(flux) | (flux == 0)

    time = time[~nancut]
    flux = flux[~nancut]
    error = np.zeros(len(time))

    if flatten:

        lcurve = np.array([time, flux, np.ones(len(time))]).T

        lcflat = tsf.TESSflatten(lcurve, split=True, winsize=winsize, stepsize=0.15, 
                                 polydeg=3, niter=10, sigmaclip=4., gapthresh=100., 
                                 tc_per=per, tc_t0=t0, tc_tdur= tdur, transitcut=True)

        flux = lcflat

    hdu.close()

    error = np.full_like(flux, futils.MAD(flux))
    
    lc = np.array([time, flux, error]).T

    return lc


