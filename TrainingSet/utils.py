import os
from pathlib import Path
from astroquery.mast import Observations, Catalogs
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.io import fits
import numpy as np
import pandas as pd
import natsort as ns
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from CandidateSet import TESSselfflatten as tsf
from Features import utils as futils
import matplotlib.pyplot as plt
from scipy.ndimage import  minimum_filter
from scipy.signal import find_peaks
import warnings
import pickle

def phasefold(time, per, t0):
    return np.mod(time - t0, per) / per


def read_parameters(sim_type, sim_location):
    sim_columns = {}
    
    sim_columns['PLA'] = {'TIC':'ticid',
                          'star.teff':'target_teff',
                          'star.mact':'target_mass',
                          'star.logg':'target_logg',
                          'star.R':'target_R',
                          'star.Tmag':'target_Tmag',
                          'star.Gmag':'target_Gmag',
                          'star.dist':'target_dist',
                          'star.L':'target_L',
                          'star.BP':'target_BP',
                          'star.RP':'target_RP',
                          'planets[0].Mp':'Mp',
                          'planets[0].Rp':'Rp',
                          'planets[0].orbital_parameters.P':'P',
                          'planets[0].orbital_parameters.ecc':'ecc',
                          'planets[0].orbital_parameters.omega':'omega',
                          'planets[0].orbital_parameters.incl':'incl',
                          'planets[0].orbital_parameters.b':'b',
                          'planets[0].ar':'ar'
                          }
    
    sim_columns['EB'] = {'TIC':'ticid',
                         'star1.teff':'target_teff',
                         'star1.mact':'target_mass',
                         'star1.logg':'target_logg',
                         'star1.R':'target_R',
                         'star1.Tmag':'target_Tmag',	
                         'star1.Gmag':'target_Gmag',
                         'star1.dist':'target_dist',
                         'star1.L':'target_L',	
                         'star1.BP':'target_BP',	
                         'star1.RP':'target_RP',
                         'star2.teff':'star2_teff',
                         'star2.mact':'star2_mass',	
                         'star2.R':'star2_R',	
                         'star2.Tmag':'star2_Tmag',
                         'star2.Gmag':'star2_Gmag',
                         'star2.L':'star2_L',	
                         'star2.BP':'star2_BP',	
                         'star2.RP':'star2_RP',	
                         'orbital_parameters.P':'P',
                         'orbital_parameters.ecc':'ecc',
                         'orbital_parameters.omega':'omega',
                         'orbital_parameters.incl':'incl',
                         'orbital_parameters.b':'b',
                         'ar':'ar'
                         }
    
    sim_columns['TRIPLE'] = {'TIC':'ticid',
                             'object1.teff':'target_teff',
                             'object1.mact':'target_mass',
                             'object1.logg':'target_logg',
                             'object1.R':'target_R',
                             'object1.Tmag':'target_Tmag',	
                             'object1.Gmag':'target_Gmag',
                             'object1.dist':'target_dist',
                             'object1.L':'target_L',	
                             'object1.BP':'target_BP',	
                             'object1.RP':'target_RP',
                             'object2.star1.teff':'star1_teff',
                             'object2.star1.mact':'star1_mass',	
                             'object2.star1.R':'star1_R',	
                             'object2.star1.Tmag':'star1_Tmag',
                             'object2.star1.Gmag':'star1_Gmag',
                             'object2.star1.L':'star1_L',	
                             'object2.star1.BP':'star1_BP',	
                             'object2.star1.RP':'star1_RP',	
                             'object2.star2.teff':'star2_teff',
                             'object2.star2.mact':'star2_mass',	
                             'object2.star2.R':'star2_R',	
                             'object2.star2.Tmag':'star2_Tmag',
                             'object2.star2.Gmag':'star2_Gmag',
                             'object2.star2.L':'star2_L',	
                             'object2.star2.BP':'star2_BP',	
                             'object2.star2.RP':'star2_RP',
                             'object2.orbital_parameters.P':'P',
                             'object2.orbital_parameters.ecc':'ecc',
                             'object2.orbital_parameters.omega':'omega',
                             'object2.orbital_parameters.incl':'incl',
                             'object2.orbital_parameters.b':'b',
                             'object2.ar':'ar'	
                             }
    
    sim_columns['PIB'] = {'TIC':'ticid',
                          'object1.teff':'target_teff',
                          'object1.mact':'target_mass',
                          'object1.logg':'target_logg',
                          'object1.R':'target_R',
                          'object1.Tmag':'target_Tmag',	
                          'object1.Gmag':'target_Gmag',
                          'object1.dist':'target_dist',
                          'object1.L':'target_L',	
                          'object1.BP':'target_BP',	
                          'object1.RP':'target_RP',
                          'object2.star.teff':'star1_teff',
                          'object2.star.mact':'star1_mass',	
                          'object2.star.R':'star1_R',	
                          'object2.star.Tmag':'star1_Tmag',
                          'object2.star.Gmag':'star1_Gmag',
                          'object2.star.L':'star1_L',	
                          'object2.star.BP':'star1_BP',	
                          'object2.star.RP':'star1_RP',
                          'object2.planets[0].Mp':'Mp',
                          'object2.planets[0].Rp':'Rp',
                          'object2.planets[0].orbital_parameters.P':'P',
                          'object2.planets[0].orbital_parameters.ecc':'ecc',
                          'object2.planets[0].orbital_parameters.omega':'omega',
                          'object2.planets[0].orbital_parameters.incl':'incl',
                          'object2.planets[0].orbital_parameters.b':'b',
                          'object2.planets[0].ar':'ar'
                          }
    
    sim_columns['BTP'] = {'TIC':'ticid',
                          '[1].teff':'target_teff',
                          '[1].mact':'target_mass',
                          '[1].logg':'target_logg',
                          '[1].R':'target_R',
                          '[1].Tmag':'target_Tmag',	
                          '[1].Gmag':'target_Gmag',
                          '[1].dist':'target_dist',
                          '[1].L':'target_L',	
                          '[1].BP':'target_BP',	
                          '[1].RP':'target_RP',
                          '[0].star.teff':'star1_teff',
                          '[0].star.mact':'star1_mass',	
                          '[0].star.R':'star1_R',	
                          '[0].star.Tmag':'star1_Tmag',
                          '[0].star.Gmag':'star1_Gmag',
                          '[0].star.dist':'star1_dist',
                          '[0].star.L':'star1_L',	
                          '[0].star.BP':'star1_BP',	
                          '[0].star.RP':'star1_RP',
                          '[0].planets[0].Mp':'Mp',
                          '[0].planets[0].Rp':'Rp',
                          '[0].planets[0].orbital_parameters.P':'P',
                          '[0].planets[0].orbital_parameters.ecc':'ecc',
                          '[0].planets[0].orbital_parameters.omega':'omega',
                          '[0].planets[0].orbital_parameters.incl':'incl',
                          '[0].planets[0].orbital_parameters.b':'b',
                          '[0].planets[0].ar':'ar'
                          }
    
    sim_columns['BEB'] = {'TIC':'ticid',
                          '[1].teff':'target_teff',
                          '[1].mact':'target_mass',
                          '[1].logg':'target_logg',
                          '[1].R':'target_R',
                          '[1].Tmag':'target_Tmag',	
                          '[1].Gmag':'target_Gmag',
                          '[1].dist':'target_dist',
                          '[1].L':'target_L',	
                          '[1].BP':'target_BP',	
                          '[1].RP':'target_RP',
                          '[0].star1.teff':'star1_teff',
                          '[0].star1.mact':'star1_mass',
                          '[0].star1.R':'star1_R',	
                          '[0].star1.Tmag':'star1_Tmag',
                          '[0].star1.Gmag':'star1_Gmag',
                          '[0].star1.dist':'star1_dist',    
                          '[0].star1.L':'star1_L',	
                          '[0].star1.BP':'star1_BP',	
                          '[0].star1.RP':'star1_RP',	
                          '[0].star2.teff':'star2_teff',
                          '[0].star2.mact':'star2_mass',	
                          '[0].star2.R':'star2_R',	
                          '[0].star2.Tmag':'star2_Tmag',
                          '[0].star2.Gmag':'star2_Gmag',
                          '[0].star2.L':'star2_L',	
                          '[0].star2.BP':'star2_BP',	
                          '[0].star2.RP':'star2_RP',
                          '[0].orbital_parameters.P':'P',
                          '[0].orbital_parameters.ecc':'ecc',
                          '[0].orbital_parameters.omega':'omega',
                          '[0].orbital_parameters.incl':'incl',
                          '[0].orbital_parameters.b':'b',
                          '[0].ar':'ar'
                          }
    
    sim_location = Path(sim_location)
    files = list(sim_location.glob(f'{sim_type}-parameters-*.csv'))
    files = ns.natsorted(files)
    
    if sim_type[0] == 'u':
        sim_type = sim_type[1:]
    
    df_lst = []
    for f in files:
        batch_num = int(f.stem.split('-')[-1])
        
        df = pd.read_csv(f)
        df = df[sim_columns[sim_type].keys()]
        df.rename(columns=sim_columns[sim_type], inplace=True)

        df.index.name = 'sim_num'
        df['sim_batch'] = batch_num
        df.set_index(['sim_batch',df.index], inplace=True)
        
        df_lst.append(df)
        
    params = pd.concat(df_lst)
    
    per = params['P'].values * 24 * 60 * 60 * u.second
    e = params['ecc'].values
    omega = params['omega'].values
    i = params['incl'].values
    
    if sim_type == 'PLA':
        Rs = params['target_R'].values*u.R_sun.to(u.m) * u.m
        Rp = params['Rp'].values*u.R_jup.to(u.m) * u.m
        Ms = params['target_mass'].values*u.M_sun.to(u.kg) * u.kg
        Mp = params['Mp'].values*u.M_jup.to(u.kg) * u.kg
    elif  sim_type == 'PIB' or sim_type == 'BTP':
        Rs = params['star1_R'].values*u.R_sun.to(u.m) * u.m
        Rp = params['Rp'].values*u.R_jup.to(u.m) * u.m
        Ms = params['star1_mass'].values*u.M_sun.to(u.kg) * u.kg
        Mp = params['Mp'].values*u.M_jup.to(u.kg) * u.kg
    elif sim_type == 'BEB' or sim_type == 'TRIPLE':
        Rs = params['star1_R'].values*u.R_sun.to(u.m) * u.m
        Rp = params['star2_R'].values*u.R_sun.to(u.m) * u.m
        Ms = params['star1_mass'].values*u.M_sun.to(u.kg) * u.kg
        Mp = params['star2_mass'].values*u.M_sun.to(u.kg) * u.kg
    elif sim_type == 'EB':
        Rs = params['target_R'].values*u.R_sun.to(u.m) * u.m
        Rp = params['star2_R'].values*u.R_sun.to(u.m) * u.m
        Ms = params['target_mass'].values*u.M_sun.to(u.kg) * u.kg
        Mp = params['star2_mass'].values*u.M_sun.to(u.kg) * u.kg
        
    tdur = theoretical_tdur(per, e, omega, i, Rs, Rp, Ms, Mp)
    col_idx = params.columns.get_loc('P') + 1
    params.insert(col_idx, 'tdur', tdur)
      
    return params


def read_sim_rejections(sim_type, sim_location):
    target_columns = {'TIC':'ticid',
                      'Target1.teff[0]':'target_teff',
                      'Target1.mact[0]':'target_mass',
                      'Target1.R[0]':'target_R',
                      'Target1.Tmag[0]':'target_Tmag',
                      'Target1.Gmag[0]':'target_Gmag',
                      'Target1.dist[0]':'target_dist'}
    
    sim_columns = {}
    
    sim_columns['PLA'] = {'Planet1.Rp[0]':'Rp',
                          'Planet1.P[0]':'P',
                          'Planet1.ecc[0]':'ecc',
                          'Planet1.omega[0]':'omega',
                          'Planet1.incl[0]':'incl',
                          'fail':'fail'
                        }
    
    sim_columns['EB'] = {'qBinary1.q[0]':'q',
                         'qBinary1.P[0]':'P',
                         'qBinary1.ecc[0]':'ecc',
                         'qBinary1.omega[0]':'omega',
                         'qBinary1.incl[0]':'incl',
                         'fail':'fail'
                        }
    
    sim_columns['TRIPLE'] = {'IsoBinary1.P[0]':'P',
                             'IsoBinary1.ecc[0]':'ecc',
                             'IsoBinary1.omega[0]':'omega',
                             'IsoBinary1.incl[0]':'incl',
                             'fail':'fail'
                            }
    
    sim_columns['PIB'] = {'Planet1.Rp[0]':'Rp',
                          'Planet1.P[0]':'P',
                          'Planet1.ecc[0]':'ecc',
                          'Planet1.omega[0]':'omega',
                          'Planet1.incl[0]':'incl',
                          'fail':'fail'
                        }
    
    sim_columns['BTP'] = {'Blend1.dist[0]':'btp_dist',
                          'Planet1.Rp[0]':'Rp',
                          'Planet1.P[0]':'P',
                          'Planet1.ecc[0]':'ecc',
                          'Planet1.omega[0]':'omega',
                          'Planet1.incl[0]':'incl',
                          'fail':'fail'
                        }
    
    sim_columns['BEB'] = {'Blend1.dist[0]':'beb_dist',
                         'IsoBinary1.P[0]':'P',
                         'IsoBinary1.ecc[0]':'ecc',
                         'IsoBinary1.omega[0]':'omega',
                         'IsoBinary1.incl[0]':'incl',
                         'fail':'fail'
                         }
    
    sim_location = Path(sim_location)
    files = list(sim_location.glob(f'{sim_type}-rejections-*.csv'))
    files = ns.natsorted(files)
    
    if sim_type[0] == 'u':
        sim_type = sim_type[1:]
        
    cols = target_columns | sim_columns[sim_type]
    
    df_lst = []
    
    for f in files:
        batch_num = int(f.stem.split('-')[-1])
        
        df = pd.read_csv(f, usecols=cols.keys())
        
        df = df[cols.keys()]
        
        df.rename(columns=cols, inplace=True)

        df.index.name = 'sim_num'
        df['sim_batch'] = batch_num
        df.set_index(['sim_batch',df.index], inplace=True)
        
        df_lst.append(df)
        
    rejections = pd.concat(df_lst)
    
    return rejections
     
    
def add_stellar_params(sim_type, params):
    params = params.copy()
    if sim_type == "PLA" or sim_type == 'uPLA':
        params["BP-RP"] = params["target_BP"] - params["target_RP"]
        params["Gmag"] = params["target_Gmag"]
        params["Tmag"] = params["target_Tmag"]
        
    elif sim_type == "EB":
        BP_RP = -2.5 * np.log10(
            10 ** (-0.4 * params["target_BP"])
            + 10 ** (-0.4 * params["star2_BP"])
        ) + 2.5 * np.log10(
            10 ** (-0.4 * params["target_RP"])
            + 10 ** (-0.4 * params["star2_RP"])
        )
        params["BP-RP"] = BP_RP
        params["Gmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["target_Gmag"])
            + 10 ** (-0.4 * params["star2_Gmag"])
        )
        params["Tmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["target_Tmag"])
            + 10 ** (-0.4 * params["star2_Tmag"])
        )

    elif sim_type == "BEB":
        BP_RP = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_BP"])
            + 10 ** (-0.4 * params["star2_BP"])
            + 10 ** (-0.4 * params["target_BP"])
        ) + 2.5 * np.log10(
            10 ** (-0.4 * params["star1_RP"])
            + 10 ** (-0.4 * params["star2_RP"])
            + 10 ** (-0.4 * params["target_RP"])
        )
        params["BP-RP"] = BP_RP
        params["Gmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Gmag"])
            + 10 ** (-0.4 * params["star2_Gmag"])
            + 10 ** (-0.4 * params["target_Gmag"])
        )
        params["Tmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Tmag"])
            + 10 ** (-0.4 * params["star2_Tmag"])
            + 10 ** (-0.4 * params["target_Tmag"])
        )

    elif sim_type == "BTP" or sim_type == 'uBTP':
        BP_RP = -2.5 * (
            np.log10(
                10 ** (-0.4 * params["star1_BP"])
                + 10 ** (-0.4 * params["target_BP"])
            )
            - np.log10(
                10 ** (-0.4 * params["star1_RP"])
                + 10 ** (-0.4 * params["target_RP"])
            )
        )
        params["BP-RP"] = BP_RP
        params["Gmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Gmag"])
            + 10 ** (-0.4 * params["target_Gmag"])
        )
        params["Tmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Tmag"])
            + 10 ** (-0.4 * params["target_Tmag"])
        )

    elif sim_type == "PIB" or sim_type == 'uPIB':
        BP_RP = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_BP"])
            + 10 ** (-0.4 * params["target_BP"])
        ) + 2.5 * np.log10(
            10 ** (-0.4 * params["star1_RP"])
            + 10 ** (-0.4 * params["target_RP"])
        )
        params["BP-RP"] = BP_RP
        params["Gmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Gmag"])
            + 10 ** (-0.4 * params["target_Gmag"])
        )
        params["Tmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Tmag"])
            + 10 ** (-0.4 * params["target_Tmag"])
        )

    elif sim_type == "TRIPLE":
        BP_RP = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_BP"])
            + 10 ** (-0.4 * params["star2_BP"])
            + 10 ** (-0.4 * params["target_BP"])
        ) + 2.5 * np.log10(
            10 ** (-0.4 * params["star1_RP"])
            + 10 ** (-0.4 * params["star2_RP"])
            + 10 ** (-0.4 * params["target_RP"])
        )
        params["BP-RP"] = BP_RP
        params["Gmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Gmag"])
            + 10 ** (-0.4 * params["star2_Gmag"])
            + 10 ** (-0.4 * params["target_Gmag"])
        )
        params["Tmag"] = -2.5 * np.log10(
            10 ** (-0.4 * params["star1_Tmag"])
            + 10 ** (-0.4 * params["star2_Tmag"])
            + 10 ** (-0.4 * params["target_Tmag"])
        )

    return params


def load_simulation(sim_dir, sim_type, sim_batch, sim_num):
    sim_dir = Path(sim_dir)
    file = sim_dir / f'{sim_type}-simu-{sim_batch}-{sim_num}.csv'

    # Load simulation data
    sim = np.genfromtxt(file)

    phase = phasefold(np.arange(len(sim)), len(sim), - len(sim) / 2) - 0.5

    sort_idx = np.argsort(phase)

    sim = sim[sort_idx]

    return sim


def load_spoc_lc(file):
    hdu = fits.open(file)
    time = hdu[1].data['TIME']
    if 'PDCSAP_FLUX' in hdu[1].columns.names:
        flux = hdu[1].data['PDCSAP_FLUX']
    else:
        flux = hdu[1].data['SAP_FLUX']
    nancut = np.isnan(time) | np.isnan(flux) | (flux == 0)

    time = time[~nancut]
    norm = np.median(flux[~nancut])
    flux = flux[~nancut] / norm
    
    cam = hdu[0].header['CAMERA']
    ccd = hdu[0].header['CCD']
    
    fraction = hdu[1].header['CROWDSAP']

    hdu.close()

    return time, flux, cam, ccd, fraction


def download_spoc_lc(ticid, sector):
    import lightkurve as lk
    ls = lk.search_lightcurve(f'TIC {ticid}', author='TESS-SPOC', sector=sector)

    if len(ls) == 0:
        return None
    else:
        lc = ls.download(quality_bitmask='none')
        
        time = lc['time'].value
        
        flux = lc['flux'].value
        
        time = time[~flux.mask]
        flux = flux[~flux.mask]
        
        flux = flux.unmasked
        
        cam = lc.meta['CAMERA']
        ccd = lc.meta['CCD']
        
        fraction = lc.meta['CROWDSAP']
        
        return time, flux, cam, ccd, fraction    
    

def load_injected_lc(file, sec_limit=[], flatten=True, winsize=4.0, transitcut=False, t0=None, per=None, tdur=None, outliers=True):
    hdu = fits.open(file)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['FLUX']
    
    seclen = [int(x) for x in hdu[0].header['SECLEN'].split(' ')]
    scc = hdu[0].header['SCC'].split(' ')
    
    sec_lcs = {}
    
    sec_lc = {}
    
    s = 0
    e = 0
    for i, length in enumerate(seclen):
        e += length
        sec_lcs[scc[i]] = (time[s:e], flux[s:e])
        s += length

    new_time = np.array([])
    new_flux = np.array([])
    new_error = np.array([])
    
    for s in scc:
        sec = int(s[1:3])
        
        if sec_limit:
            if sec < sec_limit[0] or sec > sec_limit[1]:
                continue
            
        sec_time, sec_flux = sec_lcs[s]
        nancut = np.isnan(sec_time) | np.isnan(sec_flux) | (sec_flux == 0)

        sec_time = sec_time[~nancut]
        sec_flux = sec_flux[~nancut]

        if flatten:
            lcurve = np.array([sec_time, sec_flux, np.ones(len(sec_time))]).T

            lcflat = tsf.TESSflatten(lcurve, split=True, winsize=winsize, stepsize=0.15, 
                                     polydeg=3, niter=10, sigmaclip=4., gapthresh=100., 
                                     transitcut=transitcut, tc_per=per, tc_t0=t0, tc_tdur= tdur)

            sec_flux = lcflat
        
        mask = sec_flux > 0
        
        sec_flux = sec_flux[mask]
        sec_time = sec_time[mask]
           
        flux_mad = futils.MAD(sec_flux)
        
        if outliers:
            mask = ((sec_flux - np.nanmedian(sec_flux))/flux_mad) > 4
            
            sec_flux = sec_flux[~mask]
            sec_time = sec_time[~mask]
            
            flux_mad = futils.MAD(sec_flux)
        
        sec_error = np.full_like(sec_flux, flux_mad)
        
        sec_lc[s] = {'time':sec_time,
                        'flux':sec_flux,
                        'error':sec_error}
        
        new_time = np.concatenate([new_time, sec_time])
        new_flux = np.concatenate([new_flux, sec_flux])
        new_error = np.concatenate([new_error, sec_error])

    lc = {'time':new_time,
          'flux':new_flux,
          'error':new_error}
    
    hdu.close()

    return lc, sec_lc


def calculate_depth(sim, per, tdur):
    phase = np.linspace(-0.5, 0.5, len(sim))
    
    tdur_p = tdur / per
    
    idx_transit = np.abs(phase) <= 0.5*tdur_p
    
    left = np.searchsorted(phase, -tdur_p*0.55, side='right')-1
    right = np.searchsorted(phase, tdur_p*0.55, side='right')-1

    try:
        depth = np.mean([sim[left], sim[right]]) - np.min(sim[idx_transit])
    except IndexError:
        depth = np.nan
    
    if np.isnan(depth) or depth <= 0:
        depth = np.median(sim) - sim[np.searchsorted(phase, 0)]

    return depth


def sim_depth(sim_type, sim_location, sim_batch, sim_num, per, tdur):
    try:        
        sim = load_simulation(sim_location, sim_type, sim_batch, sim_num)
        
        depth = calculate_depth(sim, per, tdur)*1e6
        
        return (sim_batch, sim_num, depth)
    except Exception as e:
        print(sim_batch, sim_num, e)
        return (sim_batch, sim_num, np.nan)


def multi_tmag_dist(ticids, workers):
    with ThreadPoolExecutor(max_workers=min(20, workers)) as ex:
        res = ex.map(tic_tmag_dist, ticids, timeout=20)
     
    tmags, dists = zip(*list(res))        
    return tmags, dists
    
    
def tic_tmag_dist(ticid):
    try:
        r = Catalogs.query_object(f'TIC {ticid}', catalog='Tic', radius=0.1*u.arcsec)
        tmag, dist = r['ID' == ticid]['Tmag', 'd']
    except Exception:
        print('Connection failed. Mangitude not retrieved.')
        return np.nan, np.nan
    
    return tmag, dist


def sim_tmag(sim_type, sim_params, target_tmag):
    if sim_type in ['PLA','BEB','NEB','NTRIPLE','NPLA']:
        sim_tmag = target_tmag
    elif sim_type == 'EB':
        l1 = sim_params['target_L']
        l2 = sim_params['star2_L']
        sim_tmag = 2.5*np.log10(l1/(l1+l2))+target_tmag
    elif sim_type == 'TRIPLE':
        l1 = sim_params['target_L']
        l2 = sim_params['star1_L'] + sim_params['star2_L']
        sim_tmag = 2.5*np.log10(l1/(l1+l2))+target_tmag
    elif sim_type == 'PIB':
        l1 = sim_params['target_L']
        l2 = sim_params['star_L']
        sim_tmag = 2.5*np.log10(l1/(l1 + l2))+target_tmag
    elif sim_type == 'BTP':
        target_dist = sim_params['dist']
        btp_dist = sim_params['dist_1']
        
        f1 = sim_params['target_L'] / (4*np.pi*target_dist*target_dist)
        f2 = sim_params['star_L'] / (4*np.pi*btp_dist*btp_dist)
    
        sim_tmag = 2.5*np.log10(f1/(f1+f2))+target_tmag
        
    return sim_tmag


def choose_epoch(time, p, tdur):
    # Calculate max_allowed time, which should be less than the period of the transits
    max_time = time[0] + p - tdur*0.5

    epoch = np.random.uniform(time[0], max_time)

    return epoch


def interpolate_sim(sim):
    # Interpolates the simulated model based on phase
    interpolation = interp1d(np.linspace(-0.5, 0.5, len(sim)), sim)

    return interpolation


def interpolate_Nmin(sim, p, exp_time):
    tpp_min = np.round((p*24*60)/len(sim))

    N_min = int(np.round(exp_time/tpp_min))

    sim_Nmin = pd.Series(sim).rolling(N_min, min_periods=1, center=True).mean().values

    func_Nmin = interpolate_sim(sim_Nmin)

    return func_Nmin, sim_Nmin


def interpolate_exposure(sim, per, exp_time, exp_unit='S'):
    sim_series = pd.Series(sim, index=pd.TimedeltaIndex(np.linspace(0, per, len(sim)), unit='D'))
    
    sim_resampled = sim_series.rolling(f'{exp_time}{exp_unit}', min_periods=1, center=True).mean().values
    
    sim_func = interpolate_sim(sim_resampled)

    return sim_func, sim_resampled


def theoretical_tdur(per, e, omega, i, Rs, Rp, Ms, Mp, primary=True):
    '''
    per: period in seconds
    e: eccentricity
    omega: in radians
    i: inclination in radias
    Rs: Radius of primary star in metres
    Rp: Radius of planet/secondary star in metres
    Ms: Mass of primary star in kg
    Mp: Mass of planet/secondary star in kg
    '''
    a = ((per**2 * const.G * (Ms + Mp)) / (4 * np.pi**2))**(1/3)
    
    if primary:
        sign = 1
    else:
        sign = -1
    
    b = (a*np.cos(i)/Rs)*((1-e**2)/(1+sign*e*np.sin(omega)))
    
    zero_array = np.full_like(np.array(per), 0)*u.s
    
    b_term = np.arcsin((Rs/a) * ((1 + Rp/Rs)**2 - b**2)**0.5 / np.sin(i), out=zero_array, where=b<(1+Rp/Rs))
    
    ecc_term = ((1-e**2)**0.5 / (1+sign*e*np.sin(omega)))
    
    tdur = (per/np.pi) * b_term.value * ecc_term

    return tdur.to(u.day).value


def secondary_t0_phase(e, omega):
    a1 = e*np.cos(omega)
    a2 = e*np.sin(omega)
    
    term1 = np.arccos(a1/np.sqrt(1-a2*a2))
    term2 = np.sqrt(1-e*e)*a1/(1-a2*a2)
    
    sec_t0 = (term1-term2)/np.pi
        
    return sec_t0

    
def get_BLS_sde(period, power, size): 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if size % 2 == 0:
            size += 1
        
        mins = minimum_filter(power, size, mode='nearest')
        trend = np.poly1d(np.polyfit(np.array(period, dtype = np.float32), mins, 3))(np.array(period, dtype = np.float32))

        power_detrended = power - trend
        mad = futils.MAD(power_detrended)

        # Calculate the SDE
        sde = power_detrended/mad

    return sde


def find_BLS_peaks(SDE, sde_lim, periods):
    dist=int(np.ceil(len(SDE)/1000))
    dist = np.max((dist, 8))
        
    peaks, meta = find_peaks(SDE, height=sde_lim, distance=dist, prominence=sde_lim-1)
    height_best_idx = meta['peak_heights'].argsort()[::-1]
    peaks = peaks[height_best_idx]
    results = []
    per_lst = []
    if len(peaks) > 0:
        results.append(peaks[0])
        per_lst.append(periods[peaks[0]])
        for p in peaks[1:]:
            ratio = np.array(per_lst)/periods[p]
            ratio_inv =  periods[p]/np.array(per_lst)
            
            if ((ratio > 0.98) & (ratio < 1.02)).any() or ((ratio > 1.98) & (ratio < 2.02)).any():
                continue
            elif ((ratio_inv > 0.98) & (ratio_inv < 1.02)).any() or ((ratio_inv > 1.98) & (ratio_inv < 2.02)).any():
                continue
            else:
                results.append(p)
                per_lst.append(periods[p])
            if len(results) == 5:
                break
    else:
        return results

    return results


def get_bls_t0(time, freq, t0_phase, tdur_phase):
    phase = (time * freq - (t0_phase+tdur_phase*0.5))
    phase -= np.floor(phase)
    
    t0_time = time[futils.find_index(phase, 0)]
    
    return t0_time

        
def check_alias(per1, per2):
    # Divide each period by the other
    div1 = per1/per2
    div2 = per2/per1

    if abs(round(div1) - div1) <= 0.05:
        return True
    elif abs(round(div2) - div2) <= 0.05:
        return True
    else:
        return False
    
        
def plot_injected_lc(lc_loc, sim_type, sim_batch, sim_num, per, t0, tdur=None, save_output=True, out_ext='png'):
    lc_loc = Path(lc_loc)

    file = lc_loc / sim_type / f'{sim_type}-{sim_batch}-{sim_num}.fits'
        
    lc, lcs = load_injected_lc(file)
    
    time = lc['time']
    flux = lc['flux']
    
    if per == 0:
        per = time[-1] - time[0]

    phase = futils.phasefold(time, per, t0 - 0.5*per) - 0.5
    idx = np.argsort(phase)
    phase = phase[idx]
    
    if tdur is not None:
        limit = np.abs(phase) < tdur*10/per
    else:
        limit = np.ones_like(phase).astype(bool)
    
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.scatter(phase[limit], flux[idx][limit], s=0.5, c='k')
    ax.set_xlabel('Phase', fontsize=14)
    ax.set_ylabel('Normalized Flux', fontsize=14)
    
    return ax


def load_training_ids(sim_type1, sim_type2, train_suffix1='training', train_suffix2='training', load_suffix='', directory=None):
    if directory is None:
        directory = Path(__file__).resolve().parent / 'Simulations'
    else:
        directory = Path(directory)
        
    output_dir1 =  directory / f'{sim_type1}'
    output_dir2 =  directory / f'{sim_type2}'
    
    sim1_train_ids = pd.read_csv(output_dir1 / f'{sim_type1}_{train_suffix1}_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
    
    sim2_train_ids = pd.read_csv(output_dir2 / f'{sim_type2}_{train_suffix2}_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
        
    # Balance training sets
    if len(sim1_train_ids) > len(sim2_train_ids):
        sim1_train_ids = sim1_train_ids.sample(len(sim2_train_ids), replace=False, random_state=10).index
        sim2_train_ids = sim2_train_ids.index
        print(f'{sim_type1} training ids downsized to {len(sim1_train_ids)}')
        print(f'Unique sims {len(sim1_train_ids.unique())}')
    elif len(sim2_train_ids) > len(sim1_train_ids):
        sim2_train_ids = sim2_train_ids.sample(len(sim1_train_ids), replace=False, random_state=10).index
        sim1_train_ids = sim1_train_ids.index
        print(f'{sim_type2} training ids downsized to {len(sim2_train_ids)}')
        print(f'Unique sims {len(sim2_train_ids.unique())}')
    else:
        sim1_train_ids = sim1_train_ids.index
        sim2_train_ids = sim2_train_ids.index
        
    return sim1_train_ids, sim2_train_ids

            
def prepare_training_sets(features, disp):
    # Get sim_type
    sim_type = features['disp'].unique()[0]
    # Drop ticid, disp and t0 from features
    features.drop(['ticid', 'disp', 't0', 'tdur', 'tdur_per', 'depth',  'transits', 
                   'median_SES', 'min_SES', 'MES_0', 'albedo', 'ptemp', 'teff', 'ptemp_stat'], axis=1, inplace=True)
    # Replace invalid values
    features.replace(np.inf, 0, inplace=True)
    features.replace("#NAME?", 0, inplace=True)
    features.fillna(0, inplace=True)
    # Ensure all types are left as numeric
    features = features.apply(pd.to_numeric)
    features.where(np.abs(features) < 1e7, 0, inplace=True)
    # Set the disposition to either 0 or 1 to use for class label training
    features['disp'] = disp
    # Add the sim_type back
    features['sim'] = sim_type
    
    return features 


def calibration_curve(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    prob_true = []
    prob_pred = []

    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_true.append(np.mean(y_true[mask]))
            prob_pred.append(np.mean(y_prob[mask]))

    return np.array(prob_pred), np.array(prob_true)


def load_trained_GP(path_to_weights):
    from TrainingSet import GPC
    from skorch.probabilistic import GPBinaryClassifier
    import torch
    
    model_weights = torch.load(path_to_weights)
    
    gp_model = GPBinaryClassifier(
    GPC.VariationalModule,
    module__inducing_points=model_weights['variational_strategy.inducing_points'], 
    criterion=torch.nn.Identity)
    
    gp_model.initialize()
    
    gp_model.module_.load_state_dict(model_weights)
    
    return gp_model


def load_trained_XGB(path_to_model):
    from xgboost import XGBClassifier
    
    xgb_model = XGBClassifier()
    
    xgb_model.load_model(path_to_model)
    
    return xgb_model


def save_pickled_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=5)
        
        
def load_pickled_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    return model