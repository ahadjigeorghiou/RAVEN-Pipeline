from Features import TransitFit
import numpy as np
from Features import utils
import astropy.units as u
import pandas as pd
from pathlib import Path


def generate_LCfeatures(lc, features, per, t0, tdur, depth, rstar, tstar, exp_time=0.02083, fill_nan=False, feat_outfile=None):
    """
    Calculate features for a given candidate or candidate list.

    Arguments:
    lc		    -- 	TBD
    				lightcurve of single candidate
    useflatten  -- 	bool
    				True to use flattened lightcurve for scatter based features
    """

    if fill_nan:
        features = utils.nan_features(features)

        return features
    
    exp_rprstar = np.sqrt(depth*1e-6)
    exp_aovrstar = (1+exp_rprstar) / np.sin(np.pi*tdur/per)
    
    fit_initialguess = np.array([per, t0, exp_aovrstar, exp_rprstar])

    transitfit = TransitFit.TransitFit(lc, fit_initialguess, exp_time=exp_time, sfactor=7)

    if not np.isnan(transitfit.params).any():
        features['fit_aovrstar'] = transitfit.params[2]
        features['fit_rprstar'] = transitfit.params[3]
        features['fit_chisq'] = transitfit.chisq
    else:
        features['fit_aovrstar'] = np.nan
        features['fit_rprstar'] = np.nan
        features['fit_chisq'] = np.nan
        
    # get duration, ingress time etc from trapezoid fit
    trapfit_initialguess = np.array([tdur * 0.9 / per, tdur / per, depth*1e-6])

    trapfit = TransitFit.TransitFit(lc, trapfit_initialguess, fittype='trap', fixper=per, fixt0=t0)
    
    if np.isnan(trapfit.params).any():
        features = utils.nan_features(features)

        return features
    
    tdur23 = trapfit.params[0]*per
    tdur = trapfit.params[1]*per
    depth = trapfit.params[2]*1e6
    
    if depth < 10:
        features = utils.nan_features(features)

        return features

    features['fit_tdur'] = tdur
    features['fit_tdur_p'] = tdur/per
    features['fit_tdur23'] = tdur23
    features['ingressdur'] = (tdur - tdur23)*0.5
    features['grazestat'] = tdur23 / tdur
    features['fit_depth'] = depth
    

    evenlc = utils.SplitOddEven(lc, per, t0, 'even')
    oddlc = utils.SplitOddEven(lc, per, t0, 'odd')

    trapfit_initialguess = np.array([tdur23/per, tdur/per, depth*1e-6])
    
    if len(oddlc['time']) != 0 and len(evenlc['time']) != 0:
        eventrapfit = TransitFit.TransitFit(evenlc, trapfit_initialguess, fittype='trap', fixper=per,
                                            fixt0=t0)
        oddtrapfit = TransitFit.TransitFit(oddlc, trapfit_initialguess, fittype='trap', fixper=per,
                                           fixt0=t0)

        features['evenodd_durratio'] = eventrapfit.params[1] / oddtrapfit.params[1]
        even_depth = eventrapfit.params[2]*1e6
        odd_depth = oddtrapfit.params[2]*1e6
        if even_depth < 1:
            even_depth = 1
        if odd_depth < 1:
            odd_depth = 1
        features['evenodd_depthratio'] = even_depth / odd_depth
    else:
        features['evenodd_durratio'] = np.nan
        features['evenodd_depthratio'] = np.nan

    if np.isnan(features['fit_rprstar']):
        rprstar = np.sqrt(depth*1e-6)
        aovrstar = (1+rprstar) / np.sin(np.pi*tdur/per)
    else:
        aovrstar = features['fit_aovrstar']
        rprstar = features['fit_rprstar']
    
    features['prad'] = utils.RPlanet(rprstar, rstar)

    phase = utils.phasefold(lc['time'], per, t0=t0 - per * 0.5) - 0.5

    features['transitSNR'] = utils.box_snr(lc['time'], lc['flux'], t0, per, tdur, depth*1e-6, phase)

    features['robstat'] = robstat(lc, per, t0, tdur, tdur23, depth)

    features = SES_metrics(features, lc, per, t0, tdur, depth*1e-6, phase)

    features = MES_metrics(features, lc, per, t0, tdur, depth*1e-6, feat_outfile)

    if not np.isnan(features['max_secmesp']):
        features, sec_depth_error = secondary_metrics(features, lc, per, t0, tdur, tdur23)
    else:
        features['sec_depth'] = np.nan
        features['sec_robstat'] = np.nan
        sec_depth_error = np.nan

    features['albedo'], features['albedo_stat'] = albedo_stat(features['sec_depth']*1e-6, sec_depth_error*1e-6, features['prad'], rstar, aovrstar)

    features['ptemp'], features['ptemp_stat'] = ptemp_stat(features['sec_depth']*1e-6, sec_depth_error*1e-6, tstar, rstar, rprstar, aovrstar)

    return features


def theoretical_depth_snr(lc, per, t0, tdur, phase=None):
    if phase is None:
        phase = utils.phasefold(lc['time'], per, t0=t0 - per * 0.5) - 0.5

    sort_idx = np.argsort(phase)

    flux = lc['flux'][sort_idx]
    phase = phase[sort_idx]

    tdur_p = (tdur / per)

    depth, depth_error = utils.calculate_depth(flux, phase, tdur_p, ignore_egress=True)

    snr = depth / depth_error

    return snr, depth*1e6


def robstat(lc, per, t0, tdur, tdur23, depth):
    flux = lc['flux']
    time = lc['time']
    error = lc['error']

    phase = utils.phasefold(time, per, t0=(t0 - per * 0.5))

    model = TransitFit.Trapezoidmodel(phase, 0.5, tdur23/per, tdur/per, depth*1e-6)

    in_transit = np.abs(phase-0.5) <= 0.5*tdur/per

    model_cut = model[in_transit]
    flux_cut = flux[in_transit]
    error_cut = error[in_transit]

    weights = np.diag(((flux_cut-model_cut)/error_cut)**2)

    rob_stat = model_cut.dot(np.matmul(weights, flux_cut)) / np.sqrt(model_cut.dot(np.matmul(weights, model_cut)))

    return rob_stat


def SES_metrics(features, lc, per, t0, tdur, depth, phase=None):
    ses = utils.SES(lc['flux'], lc['time'], tdur, depth)

    if phase is None:
        phase = utils.phasefold(lc['time'], per, t0=(t0 - per * 0.5)) - 0.5

    tdur_p = tdur/per

    idx = np.abs(phase) < tdur_p*0.5

    if sum(idx) > 0:
        split_idx = np.where(np.diff(lc['time'][idx]) > tdur)[0]
        split_idx += 1
        ses_per_transit = np.split(ses[idx], split_idx)
        max_ses_per_transit = []  
        for spt in ses_per_transit:
            max_ses_per_transit.append(np.nanmax(spt))
       
        features['max_SES'] = np.nanmax(max_ses_per_transit)
        features['median_SES'] = np.nanmedian(max_ses_per_transit)
        features['min_SES'] = np.nanmin(max_ses_per_transit)
        features['rSES_med'] = features['max_SES'] / features['median_SES']
        features['rSES_min'] = features['max_SES'] / features['min_SES']
    else:
        features['max_SES'] = np.nan
        features['median_SES'] = np.nan
        features['min_SES'] = np.nan
        features['rSES_med'] = np.nan
        features['rSES_min'] = np.nan

    return features


def MES_metrics(features, lc, per, t0, tdur, depth, output=None):
    mes, phase, square = utils.MES(lc['flux'], lc['time'], per, t0, tdur, depth)

    idx = utils.find_index(phase, 0)
    mes_0 = mes[idx]

    tdur_p = tdur/per

    idx = np.where((phase >= -tdur_p*0.5) & (phase <= tdur_p*0.5))[0]
    max_in_transit_mes = np.max(mes[idx])

    features['MES'] = max_in_transit_mes
    features['MES_0'] = mes_0
    
    # MES metrics in the absence of the primary
    s_mes, s_phase = utils.secondary_MES(lc['flux'], lc['time'], per, t0, tdur, depth, square)
    
    features['median_mes'] = np.nanmedian(s_mes)
    features['mad_mes'] = utils.MAD(s_mes)
    features['min_mes'] = np.nanmin(s_mes)
    features['rmesmed'] = features['MES'] / features['median_mes']
    features['rmesmad'] = features['MES'] / features['mad_mes']
    features['rminmes'] = features['min_mes'] / features['MES']
    features['rsnrmes'] = features['transitSNR'] / features['MES']

    # MES for secondary transit
    sec_idx = np.abs(s_phase) > 0.25
    s_mes = s_mes[sec_idx]
    s_phase = s_phase[sec_idx]
    
    if len(s_mes) > 0:
        features['max_secmes'] = np.nanmax(s_mes)
        
        maxmesp = s_phase[np.argmax(s_mes)]
        features['max_secmesp'] = maxmesp
        
        features['rsecmesmad'] = features['max_secmes']/features['mad_mes']
    else:
        features['max_secmes'] = np.nan
        features['max_secmesp'] = np.nan
        features['rsecmesmad'] = np.nan
        
    if output:
        df = pd.DataFrame({'phase': phase, 'MES': mes})
        directory = Path(output).parent.absolute()
        directory = directory / 'MES'
        if not directory.exists():
            directory.mkdir()
        outfile = directory/f'MES_{features["target"]}_{features["candidate"]}.csv'

        df.to_csv(outfile, index=False)

    return features


def secondary_metrics(features, lc, per, t0, tdur, tdur23):
    t0_sec = t0 + features['max_secmesp'] * per

    tdur_p = tdur / per
    tdur23_p = tdur23 / per

    sec_depth, sec_depth_err = utils.calculate_depth(lc['time'], lc['flux'], per, t0_sec, tdur)

    if sec_depth < 0:
        sec_depth = 1e-6
    
    initialguess = np.array([tdur23_p, tdur_p, sec_depth])
    bounds = [(tdur23_p*0.9, tdur_p*0.9, 0),(tdur23_p*1.1, tdur_p*1.1, 1)]
    trapfit = TransitFit.TransitFit(lc, initialguess, bounds=bounds, fittype='trap', fixper=per, fixt0=t0_sec)

    if np.isnan(trapfit.params).any():
        features['sec_depth'] = np.nan
        features['sec_robstat'] = np.nan
        sec_depth_err = np.nan
    else:
        sec_depth  = trapfit.params[2]*1e6
        
        if sec_depth < 10:
            sec_depth = 0
        
        sec_tdur = trapfit.params[1]*per
        sec_tdur23 = trapfit.params[0]*per
        
        sec_depth_err = trapfit.errors[2]*1e6
        
        if sec_depth_err < 1:
            sec_depth_err = 1
        
        sec_robstat = robstat(lc, per, t0_sec, sec_tdur, sec_tdur23, sec_depth)
        
        features['sec_depth'] = sec_depth
        features['sec_robstat'] = sec_robstat

    return features, sec_depth_err


def albedo_stat(sec_depth, sec_depth_err, prad, rstar, aovrstar):
    if sec_depth > 0 and not np.isnan(sec_depth):
        if not np.isnan(rstar):
            a = aovrstar * rstar * u.Rsun.to(u.Rearth)
            albedo = sec_depth * (a * a) / (prad * prad)
            albedo_err = (albedo * sec_depth_err) / sec_depth

            stat = (albedo - 1)/albedo_err
            return albedo, stat
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan


def ptemp_stat(sec_depth, sec_depth_err, tstar, rstar, rprstar, aovrstar):
    if sec_depth > 0 and not np.isnan(sec_depth):
        if not np.isnan(tstar) and not np.isnan(rstar):
            ptemp = (sec_depth ** 0.25) * tstar / np.sqrt((rprstar))

            ptemp_err = (ptemp * 0.25 * sec_depth_err) / sec_depth

            a = aovrstar * rstar

            p_eq_temp = utils.planet_eq_temp(tstar, rstar, a, bond_albedo=0.3)

            stat = (ptemp - p_eq_temp)/ptemp_err
            return ptemp, stat
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan
