import numpy as np
from scipy.signal import convolve
import pandas as pd
import astropy.units as u
from Features import TransitFit


def phasefold(time, per, t0=0):
    return np.mod(time - t0, per) / per

def SplitOddEven(lc, per, t0, oddeven):
    time = lc['time']
    flux = lc['flux']
    err = lc['error']
    phase = phasefold(time, per * 2, t0)
    if oddeven == 'even':
        split = (phase <= 0.25) | (phase > 0.75)
    else:
        split = (phase > 0.25) & (phase <= 0.75)
    splitlc = {'time': time[split], 'flux': flux[split], 'error': err[split]}
    return splitlc


def MAD(flux):
    """
    Median Average Deviation
    """
    mednorm = np.nanmedian(flux)
    return 1.4826 * np.nanmedian(np.abs(flux - mednorm))


def RPlanet(rprs, srad):
    '''
    Fitted planetary radius in earth radius. Assumes 1 solar radius host,
    unless self.target.stellar_radius was set.
    '''
    return rprs * srad * u.Rsun.to('Rearth')


def SES(flux, time, tdur, depth):
    time_diff = np.diff(time)
    gaps = np.where(time_diff > 0.5)[0]
    gaps += 1
    gaps = list(gaps)
    gaps.append(len(time))
    
    start = 0
    
    ses_arr = np.zeros_like(time)
    
    for end in gaps:
        time_seg = time[start:end]
        flux_seg = flux[start:end] - 1
        cad = np.nanmedian(np.diff(time_seg))
        transit_points = int(np.ceil(tdur/cad))     
        out_points = int(np.ceil(transit_points/2))

        square = np.ones(transit_points + 2*out_points)
        square[out_points: -out_points] -= depth
        
        mad = MAD(flux_seg)
        if mad == 0.0:
            mad = 1

        ses = convolve(flux_seg/mad, square-1, 'same') / np.sqrt(mad)
        
        ses_arr[start:end] = ses

    return ses_arr



def MES_square(phase, per, tdur, depth):
    tdur_p = tdur / per
    transit_points = len(np.where((phase >= -tdur_p * 0.5) & (phase <= tdur_p * 0.5))[0])
    if transit_points == 0:
        bins = np.arange(-0.5, 0.5+tdur_p, tdur_p)
        bins[-1] = 0.5
        transit_points = int(np.median(np.bincount(np.digitize(phase, bins)))) 
    out_points = int(np.ceil(transit_points / 2))
    square = np.ones(transit_points + 2 * out_points)
    square[out_points: -out_points] -= depth
    
    return square

                
def MES(flux, time, per, t0, tdur, depth, square=[]):
    phase = phasefold(time, per, t0=(t0 - per * 0.5)) - 0.5
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = flux[sort_idx]
    flux -= 1
    
    if len(square) == 0:
        square = MES_square(phase, per, tdur, depth)
        
    if len(square) > 0:
        mad = MAD(flux)
        
        mes = convolve(flux/mad, square-1, 'same') / np.sqrt(mad)
    else:
        mes = np.zeros_like(phase)

    return mes, phase, square


def secondary_MES(flux, time, per, t0, tdur, depth, square):
    phase = phasefold(time, per, t0=(t0 - per * 0.5)) - 0.5

    tdur_p = tdur / per

    transit_idx = np.argwhere((phase >= -tdur_p * 0.5) & (phase <= tdur_p * 0.5))

    flux = np.delete(flux, transit_idx)
    time = np.delete(time, transit_idx)

    s_mes, phase, square = MES(flux, time, per, t0, tdur, depth, square)

    return s_mes, phase


def weighted_mean(values, error):
    weights = 1 / (error * error)

    sum1 = np.sum(values * weights)

    sum2 = np.sum(weights)

    wmean = sum1 / sum2

    werror = 1 / np.sqrt(sum2)

    return wmean, werror


def find_index(array, value):
    return np.argmin(np.abs(array - value))


def transit_cut(lc, per, t0, tdur):
    if per == 0 or np.isnan(per):
        per1 = lc['time'][-1] - (t0-tdur*0.5)
        per2 = t0-tdur*0.5 - (lc['time'][0])
        per = np.max((per1, per2))
    phase = phasefold(lc['time'], per, t0 - per*0.5) - 0.5
    tdur_p = tdur/per

    intransit = np.abs(phase)<=0.5*tdur_p
    
    lc['time'] = lc['time'][~intransit]
    lc['flux'] = lc['flux'][~intransit]
    lc['error'] = lc['error'][~intransit]
    
    return lc


def calculate_depth(time, flux, per, t0, tdur, ignore_egress=True):
    phase = phasefold(time, per, t0 - 0.5*per) - 0.5
    tdur_p = tdur/per
    if ignore_egress:
        eclipse_tdur = tdur_p * 0.8
    else:
        eclipse_tdur = tdur_p

    transit_idx = np.argwhere((phase >= -eclipse_tdur * 0.5) & (phase <= eclipse_tdur * 0.5))

    transit_flux = flux[transit_idx]

    out_idx = np.argwhere(((phase >= -2 * tdur_p) & (phase <= -0.5 * tdur_p)) | ((phase >= 0.5 * tdur_p) & (phase <= 2 * tdur_p)))

    out_flux = flux[out_idx]

    depth = np.nanmedian(out_flux) - np.nanmedian(transit_flux)

    transit_error = MAD(transit_flux)/np.sqrt(len(transit_flux))
    out_error = MAD(out_flux)/np.sqrt(len(out_flux))

    depth_error = np.sqrt((out_error * out_error) + (transit_error * transit_error))

    return depth, depth_error


def planet_eq_temp(tstar, rstar, alpha, bond_albedo):
    eq_temp = tstar * np.sqrt(rstar / (2 * alpha)) * ((1 - bond_albedo) ** 1 / 4)
    return eq_temp


def no_per(time, t0, tdur):
    if (t0 > time[-1]) or (t0 < time[0]):
        new_per = False
    else:
        cndt_per1 = time[-1] - (t0-tdur*0.5)
        cndt_per2 = t0 - (time[0]-tdur*0.5)
        new_per = np.max((cndt_per1, cndt_per2))
    
    return new_per
    
def check_transit_points(time, per, t0, tdur):
    if per == 0 or np.isnan(per):
        per = no_per(time, t0, tdur)
    
    if per:
        phase = phasefold(time, per, t0=(t0 - per * 0.5)) - 0.5

        tdur_p = tdur / per

        idx = np.where((phase >= -tdur_p*0.5) & (phase <= tdur_p*0.5))[0]

        if len(idx) == 0:
            return False
        else:
            return True
    else:
        return False


def observed_transits(time, t0, per, tdur):
    '''Returns the number of observed transits within a time window'''
    count = 0
    
    if per == 0 or np.isnan(per):
        per = no_per(time, t0, tdur)
    
    if per:
        phase = phasefold(time, per, t0=(t0 - per * 0.5)) - 0.5

        tdur_p = tdur / per
        
        idx = np.abs(phase) < 0.5*tdur_p
        
        if sum(idx) > 0:
            count = 1
            
            time_diff = np.diff(time[idx])
            transits = sum(time_diff > np.floor(per)*0.8)

            count += transits

    return count


def box_snr(time, flux, t0, per, tdur, depth, phase=None):
    if phase is None:
        phase = phasefold(time, per, t0 - 0.5*per) - 0.5
    tdur_p = tdur/per
    oot_flux = flux[np.abs(phase) > (tdur_p*0.5)]
    it_flux = flux[np.abs(phase) <= (tdur_p*0.5)]
    mad_flux = MAD(oot_flux)
    
    snr = (depth/mad_flux)*np.sqrt(len(it_flux))
    
    return snr


def box_snr2(depth, error, tdur, n_transits):
    snr = (depth/error)*np.sqrt(n_transits*tdur)
    
    return snr


def trapezoid_snr(depth, error, tdur, tdur2_3, n_transits):
    snr = (depth/error)*np.sqrt(n_transits*(tdur + 2*tdur2_3)/3)

    return snr


def trapfit_results(t0, tdur, depth, per, lc):
    if depth > 1.0:
        depth *= 1e-6
        
    trapfit_initialguess = np.array([tdur * 0.9 / per, tdur / per, depth])
    trapfit = TransitFit.TransitFit(lc, trapfit_initialguess, fittype='trap', fixper=per, fixt0=t0)
    
    fit_tdur = trapfit.params[1]*per
    fit_tdur23 = trapfit.params[0]*per
    fit_depth = trapfit.params[2]*1e6
    return fit_tdur, fit_tdur23, fit_depth


def transit_params(lc, t0, per, tdur, depth):
    transits = observed_transits(lc['time'], t0, per, tdur)
    phase = phasefold(lc['time'], per, t0 - 0.5*per) -0.5
    intransit = np.abs(phase) < 0.5*tdur/per
    
    if transits > 0 and sum(intransit) > 3:
        fit_tdur, fit_tdur23, fit_depth = trapfit_results(t0, tdur, depth, per, lc)
        
        return fit_tdur, fit_tdur23, fit_depth
    else:
        return tdur, np.nan, np.nan


def nan_features(features):
    cols = ['fit_aovrstar', 'fit_rprstar',
       'fit_chisq', 'fit_tdur', 'fit_tdur_p', 'fit_tdur23', 'ingressdur', 'grazestat',
       'fit_depth', 'evenodd_durratio', 'evenodd_depthratio', 'prad',
       'transitSNR', 'robstat', 'max_SES', 'median_SES', 'min_SES', 'rSES_med',
       'rSES_min', 'MES', 'MES_0', 'median_mes', 'mad_mes', 'min_mes', 'rmesmed',
       'rmesmad', 'rminmes', 'rsnrmes', 'max_secmes', 'max_secmesp',
       'rsecmesmad', 'sec_depth', 'sec_robstat', 'albedo', 'albedo_stat',
       'ptemp', 'ptemp_stat']

    for col in cols:
        features[col] = np.nan
        
    return features


def prepare_features(features):
    features.reset_index(inplace=True)
    # Drop not needed data
    features.drop(['ticid', 'candidate', 'disp', 'flag', 't0', 'tdur', 'tdur_per','depth',  'transits', 
                   'median_SES', 'min_SES', 'MES_0', 'albedo', 'ptemp', 'teff', 'ptemp_stat'], axis=1, inplace=True)
    # Replace invalid values
    features.replace(np.inf, 0, inplace=True)
    features.replace("#NAME?", 0, inplace=True)
    features.fillna(0, inplace=True)
    # Ensure all types are left as numeric
    features = features.apply(pd.to_numeric)
    features.where(np.abs(features) < 1e7, 0, inplace=True)
    
    return features


def bin_phasefolded_lc(phase, flux, n_bins=200):
    bins = np.linspace(np.min(phase), np.max(phase), n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    from scipy.stats import binned_statistic
    
    bin_means = binned_statistic(phase, flux, statistic='mean', bins=bins)[0]
    
    return bin_centers, bin_means