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

import os
from pathlib import Path
import sys
import numpy as np
from types import *
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from Features.TransitSOM import somutils, somtools, selfsom
except ImportError:
    print('Accompanying libraries not in PYTHONPATH or current directory')
    sys.exit()

def PrepareLightcurves(filelist,periods,t0s,tdurs,nbins=50,clip_outliers=5, multiprocessing=1):
    """
        Takes a list of lightcurve files and produces a SOM array suitable for classification or training. Single lightcurves should use PrepareOneLightcurve().
    
            Args:
                filelist: List of lightcurve files. Files should be in format time (days), flux, error. 
                periods: Array of orbital periods in days, one per input file.
                t0s: Array of epochs in days, one per input file
                tdurs: Transit durations (T14) in days, one per input file. It is important these are calculated directly from the transit - small errors due to ignoring eccentricity for example will cause bad classifications. 
                nbins: Number of bins to make across 3 transit duration window centred on transit. Empty bins will be interpolated linearly. (int)
                clip_outliers: if non-zero, data more than clip_outliers*datapoint error from the bin mean will be ignored.
                
            Returns:
                SOMarray: Array of binned normalised lightcurves
                SOMarray_errors: Array of binned lightcurve errors
                som_ids: The index in filelist of each lightcurve file used (files with no transit data are ignored)
    """
    try:
        assert len(filelist)==len(periods)==len(t0s)==len(tdurs)
    except AssertionError:
        print('Filelist, periods, epochs and transit duration arrays must be 1D arrays or lists of the same size')
        return 0,0,0
    
    try:
        assert nbins>0
        assert isinstance(nbins, int)
    except AssertionError:
        print('nbins must be positive integer')
        return 0,0,0
    
    if multiprocessing == 1:
        multiprocessing = os.cpu_count()

    SOMarray_bins = []
    som_ids = []

    dataset_size = len(filelist)
    if multiprocessing > 1 and dataset_size > multiprocessing:
        with ProcessPoolExecutor(max_workers=multiprocessing) as ex:
            try:
                factor = 20
                while dataset_size < 5*factor*multiprocessing:
                    factor -= 1
                    if factor == 1:
                        break
                idx_split = np.array_split(np.arange(len(filelist), dtype=int), factor*multiprocessing)
                
                futures = {ex.submit(multi_prepare, filelist[idx], periods[idx], t0s[idx], tdurs[idx], nbins, clip_outliers): idx for idx in idx_split}
                
                for future in as_completed(futures):
                    idx_group = futures[future]
                    try:
                        results = future.result()
                        for i,r in enumerate(results):
                            if r is not None:
                                SOMarray_bins.append(r)
                                som_ids.append(idx_group[i])
                            else:
                                print(f'Error when preparing lc for idx {idx_group[i]}')
                    except Exception as e:
                        print(f'Exception {e} occured for group: {idx_group}')
                    
            except KeyboardInterrupt:
                ex.shutdown(wait=False, cancel_futures=True)
                raise ValueError('Keyboard interrupt')
        
    else:
        for i in range(len(filelist)):
            try:
                SOMtransit_bin = PrepareOneLightcurve(filelist[i], periods[i], t0s[i], tdurs[i], nbins, clip_outliers)
                SOMarray_bins.append(SOMtransit_bin)
                som_ids.append(i)
            except Exception as e:
                print(f'Exception {e} when preparing lc for idx {i}')
          
    SOMarray = np.array(SOMarray_bins)
    som_ids = np.array(som_ids)
    sort_idx = np.argsort(som_ids)
    som_ids = som_ids[sort_idx]
    SOMarray = SOMarray[sort_idx]
    
    #SOMarray_errors = np.array(SOMarray_binerrors)
    return SOMarray, np.array(som_ids)


def multi_prepare(filepaths, pers, t0s, tdurs, nbins, clip_outliers):
    result = []
    for i in range(len(filepaths)):
        try:
            result.append(PrepareOneLightcurve(filepaths[i], pers[i], t0s[i], tdurs[i], nbins=nbins, clip_outliers=clip_outliers))
        except Exception:
            result.append(None)
    return result   
            
def PrepareOneLightcurve(filepath,per,t0,tdur,nbins=50,clip_outliers=5,lc=None):
    """
        Takes one lightcurve array and bins it to format suitable for classification.
    
            Args:
                lc: Lightcurve array. Columns should be time (days), flux, error. Nans should be removed prior to calling function. 
                per: Orbital period in days (float)
                t0: Epoch in days (float)
                tdur: Transit duration (T14) in days (float). It is important this is calculated directly from the transit - small errors due to ignoring eccentricity for example will cause bad classifications. 
                nbins: Number of bins to make across 3 transit duration window centred on transit. Empty bins will be interpolated linearly. (int)
                clip_outliers: if non-zero, data more than clip_outliers*datapoint error from the bin mean will be ignored.
                
            Returns:
                SOMtransit_bin: Binned normalised lightcurve
    """

    try:
        assert nbins>0
        assert isinstance(nbins, int)
    except AssertionError:
        print('nbins must be positive integer')

    if lc is None:
        lc = somutils.load_injected(filepath, flatten=True, winsize=2.0, t0=t0, per=per, tdur=tdur)
    #phase fold (transit at 0.5)
    phase = somutils.phasefold(lc[:,0],per,t0-per*0.5)
    idx = np.argsort(phase)
    lc = lc[idx,:]
    phase = phase[idx]

    #cut to relevant region
    tdur_phase = tdur/per
    lowidx = np.searchsorted(phase,0.5-tdur_phase*1.5)
    highidx = np.searchsorted(phase,0.5+tdur_phase*1.5)
    lc = lc[lowidx:highidx,:]
    phase = phase[lowidx:highidx]
    bin_edges = np.linspace(0.5-tdur_phase*1.5,0.5+tdur_phase*1.5,nbins+1)
    bin_edges[-1]+=0.0001                               #avoids edge problems
    
    #perform binning
    if len(lc[:,0]) != 0:
        binphases,SOMtransit_bin,binerrors = somutils.GetBinnedVals(phase,lc[:,1],lc[:,2],lc[:,2],nbins,bin_edges,clip_outliers=clip_outliers)
    else:
        return np.ones(48)

    #normalise arrays, and interpolate nans where necessary
    SOMarray_single,SOMarray_errors_single = somutils.PrepareArray(SOMtransit_bin,binerrors)

    return SOMarray_single

    
    
def CreateSOM(SOMarray,niter=500,learningrate=0.1,learningradius=None,somshape=(20,20),outfile=None, return_som=True):
    """
        Trains a SOM, using an array of pre-prepared lightcurves. Can save the SOM to text file.
        Saved SOM can be reloaded using LoadSOM() function.
    
            Args:
                SOMarray: Array of normalised inputs (e.g. binned transits), of shape [n_inputs,n_bins]. n_inputs > 1
                niter: number of training iterations, default 500. Must be positive integer.
                learningrate: alpha parameter, default 0.1. Must be positive.
                learningradius: sigma parameter, default the largest input SOM dimension. Must be positive.
                somshape: shape of SOM to train, default (20,20). Currently must be 2 dimensional tuple (int, int). Need not be square.
                outfile: File path to save SOM Kohonen Layer to. If None will not save.
    
            Returns:
                The trained SOM object
    """           
    try:
        assert niter >= 1, 'niter must be >= 1.'
        assert isinstance(niter, int), 'niter must be integer.'
        assert learningrate > 0, 'learningrate must be positive.'
        if learningradius:
            assert learningradius > 0, 'learningradius must be positive.'
        assert len(somshape)==2, 'SOM must have 2 dimensions.'
        assert (isinstance(somshape[0], int) and isinstance(somshape[1], int)), 'somshape must contain integers.'
        assert len(SOMarray.shape)==2, 'Input array must be 2D of shape [ninputs, nbins].'
        assert SOMarray.shape[0]>1, 'ninputs must be greater than 1.'
    except AssertionError as error:
        print(error)
        print('Inputs do not meet requirements. See help')
        return 0
        
    nbins = SOMarray.shape[1]
    
    #default learning radius
    if not learningradius:
        learningradius = np.max(somshape)
    
    #define som initialisation function
    def Init(sample):
        return np.random.uniform(0,2,size=(somshape[0],somshape[1],nbins))
    
    #initialise som
    som = selfsom.SimpleSOMMapper(somshape,niter,initialization_func=Init,learning_rate=learningrate,iradius=learningradius)

    #train som
    som.train(SOMarray)
    
    #save
    if outfile:
        somtools.KohonenSave(som.K,outfile)
    
    #return trained som
    if return_som:
        return som


def LoadSOM(filepath,dim_x,dim_y,nbins,lrate=0.1):
    """
        Makes a som object using a saved Kohonen Layer (such as could be saved by CreateSOM().
    
            Args:
                filepath: The path to the saved Kohonen Layer. Must be saved in format created by somtools.KohonenSave().
                dim_x: The size of the first SOM dimension. Int
                dim_y: The size of the second SOM dimension. Int
                nbins: The number of lightcurve bins used (i.e. the 3rd dimension of the Kohonen Layer). Int
                lrate: The learning rate used to train the SOM. Optional, default=0.1. Included for tidiness, 
                       if the SOM is not retrained and only used for classification this parameter does not matter.
            
            Returns:
                The SOM object
    """    
    def Init(sample):
        return np.random.uniform(0,2,size=(int(dim_x),int(dim_y),int(nbins)))
        
    som = selfsom.SimpleSOMMapper((dim_x,dim_y),1,initialization_func=Init,learning_rate=lrate)
    loadk = somtools.KohonenLoad(filepath)
    som.train(loadk) #tricks the som into thinking it's been trained
    som._K = loadk  #loads the actual Kohonen layer into place.
    return som


def generate_proportions(som, training_array, som_groups, map_shape=(20,20), outfile=None):
    """
    Generates and saves the proportions map from a trained SOM and training data.

        Args:
            som: The trained SOM object.
            training_array: The array of training data.
            som_groups: The group labels for the training data.
            outfile: The path to save the proportions numpy array.
    """
    map_training = som(training_array)
    
    proportions = somtools.Proportions_2group(map_training, som_groups, map_shape)
    
    if outfile is not None:
        np.save(outfile, proportions)
    
    return proportions


def classify(som, proportions, classify_array):
    probs = np.zeros(len(classify_array))
    map_classify = som(classify_array)
    
    for i in range(proportions.shape[0]):
        for j in range(proportions.shape[1]):
            mask = (map_classify==np.array([i,j])).all(axis=1)
            probs[mask] = proportions[i,j]
            
    distances = np.sqrt(np.sum(np.power((classify_array-som.K[map_classify[:,0], map_classify[:,1]]),2), axis=1))        
            
    return probs, distances