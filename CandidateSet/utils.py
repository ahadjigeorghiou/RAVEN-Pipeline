import astropy.constants as const
import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astroquery.mast import Catalogs, Observations
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astropy.io import fits
from scipy import stats
from . import TESSselfflatten as tsf
import numpy as np
import pickle
from pathlib import Path
import textwrap
from requests.exceptions import ConnectionError
from Features import utils as futils
import matplotlib.pyplot as plt


Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1


def lc_filepath(lc_dir, dir_style, ticid, sector):
    filename = f'hlsp_tess-spoc_tess_phot_{ticid:016}-s{sector:04}_tess_v1_lc.fits'
    if dir_style == 'per_target':
        filepath = lc_dir / f'{ticid}' / filename
    elif dir_style == 'spoc':
        full_ticid = f'{ticid:016}'
        split_id = textwrap.wrap(full_ticid, 4)
        filepath = lc_dir / F'S{sector:02}' / 'target' / split_id[0] / split_id[1] / split_id[2] / split_id[3] / filename
    elif dir_style == 'single':
        filepath = lc_dir / filename
    else:
        raise ValueError(f'dir_style: {dir_style} not supported.')
    
    return filepath


def download_spoc_lc(ticid, sec, download_dir):
    result = False
    try:
        obs = Observations.query_criteria(obs_collection=('HLSP'),
                                          provenance_name=('TESS-SPOC'),
                                          objectname=f'TIC {ticid}',
                                          radius=.001 * u.arcsec,
                                          t_exptime=(121, 1801),
                                          sequence_number=sec,
                                          target_name=ticid)

        if int(obs['target_name'][0]) == ticid:
            product = obs['dataURL'][0]
            
            if product.split('_')[-1] == 'lc.fits':                   
                download_dir.mkdir(exist_ok=True, parents=True)
                                        
                filename = product.split('/')[-1]
                            
                result = Observations.download_file(product, local_path=(download_dir / filename))[0]
                if result[0] == 'COMPLETE':
                    result = True
    except ConnectionError:
        print(ticid, 'Connection error. Observations not retrieved.')

    return result


def download_spoc_lcs(ticid, download_dir, max_sec=None):
    download_dir = Path(download_dir)
    try:
        obs = Observations.query_criteria(obs_collection=('HLSP'),
                                          provenance_name=('TESS-SPOC'),
                                          objectname=f'TIC {ticid}',
                                          radius=.001 * u.arcsec,
                                          t_exptime=(121, 1801),
                                          target_name=ticid)

        obs = obs[obs['target_name'] == str(ticid)]
        
        if max_sec is not None:
            obs = obs[obs['sequence_number'] <= max_sec]
        
        for product in obs['dataURL']:
            if product.split('_')[-1] == 'lc.fits':                   
                download_dir.mkdir(exist_ok=True, parents=True)
                                        
                filename = product.split('/')[-1]
                            
                Observations.download_file(product, local_path=(download_dir / filename))
    except ConnectionError:
        print(ticid, 'Connection error. Observations not retrieved.')
    
    
def load_default(infile, per_lim=[0, None], depth_lim=None):    
    # Default data should be a csv file with ticid, toi/candidate num, per, t0, tdur, depth in this order
    data = pd.read_csv(f'Input/{infile}')
    data.columns = ['ticid', 'candidate', 'per', 't0', 'tdur', 'depth']
    data.set_index(['ticid', 'candidate'], inplace=True)
    
    data.loc[data['t0'] > 2457000, 't0'] -= 2457000
    
    data.sort_values(['ticid', 'candidate'])
    
    data.query((f'per >= {per_lim[0]}'), inplace=True)
    
    if per_lim[1] is not None:
        data.query((f'per <= {per_lim[1]}'), inplace=True)
    
    if depth_lim is not None:
        data.query((f'depth >= {depth_lim}'), inplace=True)

    return data


def load_recovery(infile, per_lim=[0, None], depth_lim=None):
    data = pd.read_csv(f'Input/{infile}')
    
    data.rename(columns={'peak_sig':'candidate',
                         'peak_per':'per',
                         'peak_t0':'t0',
                         'peak_tdur':'tdur',
                         'peak_depth':'depth'}, inplace=True)
    
    data.set_index(['ticid', 'candidate'], inplace=True)
    
    data = data[['per', 't0', 'tdur', 'depth']]
    
    data.loc[data['t0'] > 2457000, 't0'] -= 2457000
    
    data.sort_values(['ticid', 'candidate'])
    
    data.query((f'per >= {per_lim[0]}'), inplace=True)
    
    if per_lim[1] is not None:
        data.query((f'per <= {per_lim[1]}'), inplace=True)
    
    if depth_lim is not None:
        data.query((f'depth >= {depth_lim}'), inplace=True)

    return data

def load_archive_toi(infile, per_lim=[0,None], depth_lim=None):        
    cols = ['toi', 'tid', 'tfopwg_disp', 'pl_tranmid', 'pl_orbper', 'pl_trandurh', 'pl_trandep']
    
    input_loc = Path(__file__).resolve().parents[1] / 'Input'
    
    try:
        toi_df = pd.read_csv(input_loc / infile, usecols=cols)
    except ValueError:
        toi_df = pd.read_csv(input_loc / infile, usecols=cols, comment='#')
    
    toi_df.rename(columns={'toi':'candidate',
                           'tid':'ticid',
                           'tfopwg_disp':'tfop_disp',
                           'pl_tranmid':'t0',
                           'pl_orbper':'per',
                           'pl_trandurh':'tdur',
                           'pl_trandep':'depth'}, inplace=True)
    
    toi_df['t0'] -= 2457000
    
    toi_df['tdur'] /= 24
    
    toi_df.loc[np.isnan(toi_df['per']), 'per'] = 0
    
    toi_df.set_index(['ticid', 'candidate'], inplace=True)
    
    toi_df.sort_values(['ticid', 'candidate'])
    
    toi_df.query((f'per >= {per_lim[0]}'), inplace=True)
    
    if per_lim[1] is not None:
        toi_df.query((f'per <= {per_lim[1]}'), inplace=True)
    
    if depth_lim is not None:
        toi_df.query((f'depth >= {depth_lim}'), inplace=True)
        
    return toi_df


def load_exofop_toi(infile, per_lim=[0, None], depth_lim=None):  
    cols = ['TIC ID', 'TOI', 'TESS Disposition', 'TFOPWG Disposition', 'Transit Epoch (BJD)', 'Period (days)', 'Duration (hours)', 'Depth (ppm)']
    
    input_loc = Path(__file__).resolve().parents[1] / 'Input'
    
    try:
        toi_df = pd.read_csv(input_loc / infile, usecols=cols)
    except ValueError:
        toi_df = pd.read_csv(input_loc / infile, usecols=cols, skiprows=2)
    
    toi_df.rename(columns={'TIC ID':'ticid',
                           'TOI':'candidate',
                           'TESS Disposition':'tess_disp',
                           'TFOPWG Disposition':'tfop_disp',
                           'Transit Epoch (BJD)':'t0',
                           'Period (days)':'per',
                           'Duration (hours)':'tdur',
                           'Depth (ppm)':'depth'}, inplace=True)
    
    toi_df['t0'] -= 2457000
    
    toi_df['tdur'] /= 24
    
    toi_df.set_index(['ticid', 'candidate'], inplace=True)
    
    toi_df.sort_values(['ticid', 'candidate'])
    
    toi_df.query((f'per >= {per_lim[0]}'), inplace=True)
    
    if per_lim[1] is not None:
        toi_df.query((f'per <= {per_lim[1]}'), inplace=True)
    
    if depth_lim is not None:
        toi_df.query((f'depth >= {depth_lim}'), inplace=True)
        
    return toi_df


def process_dataframe_input(data, per_lim=[0, None], depth_lim=None):
    data.reset_index(inplace=True)
    
    if 'disp' in data.columns:
        expected_columns = ['ticid', 'candidate', 'per', 't0', 'tdur', 'depth', 'disp']
    elif 'tfop_disp' in data.columns:
        expected_columns = ['ticid', 'candidate', 'per', 't0', 'tdur', 'depth', 'tfop_disp']
    else:
        expected_columns = ['ticid', 'candidate', 'per', 't0', 'tdur', 'depth']
    
    try:
        data = data[expected_columns]
    except KeyError:
        raise KeyError(f'Input dataframe must contain the following columns: {expected_columns}')
    
    data.set_index(['ticid', 'candidate'], inplace=True)
    
    data.loc[data['t0'] > 2457000, 't0'] -= 2457000
    
    data.sort_values(['ticid', 'candidate'])
    
    data.query((f'per >= {per_lim[0]}'), inplace=True)
    
    if per_lim[1] is not None:
        data.query((f'per <= {per_lim[1]}'), inplace=True)
    
    if depth_lim is not None:
        data.query((f'depth >= {depth_lim}'), inplace=True)

    return data


def query_tic_by_id(ticid, radius):
    try:
        tic_res = Catalogs.query_object(f'TIC {ticid}', catalog='Tic', radius=radius)
        tic_res = tic_res['ID', 'GAIA', 'ra', 'dec', 'RA_orig', 'Dec_orig', 'pmRA','pmDEC', 'd', 'rad', 'mass', 'Teff', 
                          'Tmag', 'GAIAmag', 'disposition', 'objType']
        tic_res = tic_res.to_pandas()
        idx = pd.notna(tic_res[['RA_orig', 'Dec_orig']]).any(axis=1)
        tic_res.loc[idx, 'ra'] = tic_res.loc[idx, 'RA_orig']
        tic_res.loc[idx, 'dec'] = tic_res.loc[idx, 'Dec_orig']
        tic_res.drop(['RA_orig', 'Dec_orig'], axis=1, inplace=True)
        tic_res.rename(columns={'ID':'ticid', 'GAIA':'Gaia_id', 'pmRA':'pmra', 'pmDEC':'pmdec',
                                'd':'dist', 'GAIAmag':'Gmag'}, inplace=True)
        tic_res['ticid'] = tic_res['ticid'].astype('Int64')
        
        return tic_res
    except ConnectionError:
        print('Connection error')
        return None
    
    
def query_gaia_dr3(ra, dec, radius):
    """
    Query the GAIA DR3 catalog by a TIC ID
    """
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
    
    try:
        job = Gaia.cone_search(coords, radius=radius)
        
        dr3 = job.get_results()

        dr3 = dr3['SOURCE_ID', 'parallax', 'phot_g_mean_mag', 'bp_rp', 'teff_gspphot', 'logg_gspphot', 'ruwe']
        
        dr3 = dr3.to_pandas()
        dr3.rename(columns={'SOURCE_ID':'Gaia_id',
                    'parallax':'Plx',
                    'phot_g_mean_mag':'Gmag',
                    'bp_rp':'BP-RP',
                    'teff_gspphot':'Teff',
                    'logg_gspphot':'logg',
                    'ruwe':'RUWE'}, inplace=True)
        
        dr3.set_index('Gaia_id', inplace=True)

        return dr3
    except Exception:
        print('Connection error')
        return pd.DataFrame()
    
    
def query_Gaia_by_coords(ticid, ra, dec, radius):
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
    try:
        dr3 = Vizier.query_region(coords, radius=radius, catalog="I/355/gaiadr3")[0]
        
        dr3 = dr3['Source', 'Plx', 'Gmag', 'BP-RP', 'Teff', 'logg', 'RUWE']
        dr3 = dr3.to_pandas()
        
        dr3.rename(columns={'Source':'Gaia_id'}, inplace=True)
        
        dr3.set_index('Gaia_id', inplace=True)

        return dr3
         
    except Exception as e:
        print(f'Exception: {e} occurred for {ticid}. No Gaia results retrieved')
        return pd.DataFrame()
    
def gaia_pm_corr(time, ra, dec, pmra, pmdec, gaia_release='dr2'):
    if gaia_release == 'dr2':
        jd = 2457206 - 2457000
    else:
        jd = 2457388.5 - 2457000
    
    time_elapsed = ((time - jd)*u.day).to(u.year)
    
    cor_ra = ra + (((pmra*u.milliarcsecond/u.year)*time_elapsed).to(u.degree)).value
    cor_dec = dec + (((pmdec*u.milliarcsecond/u.year)*time_elapsed).to(u.degree)).value
    
    return cor_ra, cor_dec


def remove_targets_from_pipeline_output(ticids, load_suffix, save_suffix):
    with open(f'Output/sources_{load_suffix}.pkl', 'rb') as f:
        sources = pickle.load(f)

    for tic in ticids:
        try:
            del sources[tic]
        except KeyError:
            continue
        
    with open(f'Output/sources_{save_suffix}.pkl', 'wb') as f:
        pickle.dump(sources, f)
        
    sector_data = pd.read_csv(f'Output/sectordata_{load_suffix}.csv', index_col=['ticid'])
    sector_data = sector_data.loc[sector_data.index.difference(ticids)]
    sector_data.to_csv(f'Output/sectordata_{save_suffix}.csv')
        
    centroid_data = pd.read_csv(f'Output/centroiddata_{load_suffix}.csv', index_col=['ticid'])
    centroid_data = centroid_data.loc[centroid_data.index.difference(ticids)]
    centroid_data.to_csv(f'Output/centroiddata_{save_suffix}.csv')
    
    probabilities = pd.read_csv(f'Output/Probabilities_{load_suffix}.csv', index_col=['target'])
    probabilities = probabilities.loc[probabilities.index.difference(ticids)]
    probabilities.to_csv(f'Output/Probabilities_{save_suffix}.csv')
    
    priors = pd.read_csv(f'Output/Priors_{load_suffix}.csv', index_col=['ticid'])
    priors = priors.loc[priors.index.difference(ticids)]
    priors.to_csv(f'Output/Priors_{save_suffix}.csv')
    
    features = pd.read_csv(f'Output/Features_{load_suffix}.csv', index_col=['ticid'])
    som_array = np.load(f'Output/SOM_array_{load_suffix}.npy')
    features['idx'] = np.arange(len(features), dtype=int)
    features = features.loc[features.index.difference(ticids)]
    som_array = som_array[features['idx'].values]
    features.drop('idx', axis=1, inplace=True)
    features.to_csv(f'Output/Features_{save_suffix}.csv')
    np.save(f'Output/SOM_array_{save_suffix}.npy', som_array)
    

def optimise_multi(datanum, workers):
    if datanum < workers:
        workers = datanum

    factor = 20
    while datanum < 5*factor*workers:
        factor -= 1
        if factor == 1:
            break
    
    return workers, datanum


def load_spoc_lc(filepath, hdu=None, flatten=False, winsize=2.0, transitcut=False, tc_per=None, tc_t0=None, tc_tdur=None, return_trend=False, return_hdu=False):
    """
    Loads a TESS lightcurve, normalised with NaNs removed.
 
    Returns:
    lc -- 	dict
 		Lightcurve with keys time, flux, error. Error is populated with zeros.
    """
    if hdu is None:
        hdu = fits.open(filepath)
    time = hdu[1].data['TIME']

    if 'PDCSAP_FLUX' in hdu[1].columns.names:
        flux = hdu[1].data['PDCSAP_FLUX']
        err = hdu[1].data['PDCSAP_FLUX_ERR']
    else:
        flux = hdu[1].data['SAP_FLUX']
        err = np.zeros(len(flux))
    nancut = np.isnan(time) | np.isnan(flux) | (flux == 0)
    lc = {}
    lc['time'] = time[~nancut]
    lc['flux'] = flux[~nancut]
    lc['error'] = err[~nancut]
     
    norm = np.median(lc['flux'])
    lc['median'] = norm
           
    lcurve = np.array([lc['time'], lc['flux'], np.ones(len(lc['time']))]).T
    
    lcflat, trend = tsf.TESSflatten(lcurve, split=True, winsize=winsize, stepsize=0.15, polydeg=3,
                                niter=10, sigmaclip=4., gapthresh=100., transitcut=transitcut,
                                tc_per=tc_per, tc_t0=tc_t0, tc_tdur=tc_tdur, return_trend=True)
      
    if flatten:    
        lc['flux'] = lcflat
         
        mad = futils.MAD(lc['flux'])
        
        if isinstance(tc_per, np.ndarray) or isinstance(tc_per, list):
            intransit = np.zeros(len(lc['time']), dtype=bool)
            for i, p in enumerate(tc_per):
                if p == 0:
                    p1 = lc['time'][-1] - (tc_t0[i]-tc_tdur[i]*0.5)
                    p2 = tc_t0[i] - (lc['time'][0]-tc_tdur[i]*0.5)
                    p = np.max((p1, p2))
                normphase = np.abs((np.mod(lc['time']-tc_t0[i]-p*0.5, p) - 0.5*p) / (0.5*tc_tdur[i]))
                intransit += normphase <= 1
        elif tc_per is not None:
            normphase = np.abs((np.mod(lc['time']-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
            intransit = normphase <= 1
        else:
            intransit = np.full_like(lc['time'], 0)
        
        # Remove intransit upward outliers
        outliers = (lc['flux'] - np.median(lc['flux']))/mad > 4
        # outliers[~intransit] = False
                        
        lc['time'] = lc['time'][~outliers]
        lc['flux'] = lc['flux'][~outliers]
        lc['error'] = lc['error'][~outliers]
        
    else:
        lc['flux'] = lc['flux']/norm
    
    trend = trend/norm
    lc['error'] = np.full_like(lc['flux'], futils.MAD(lc['flux']))

    if return_hdu and return_trend:
        return lc, trend, hdu
    elif return_hdu and not return_trend:
        return lc, hdu
    elif not return_hdu and return_trend:
        hdu.close()
        del hdu
        
        return lc, trend
    else:
        hdu.close()
        del hdu
        
        return lc
         
         
def load_spoc_centroid(filepath, flatten=False, trim=False, cut_outliers=False, transitcut=False, tc_per=None, tc_t0=None, tc_tdur=None):
    """
    Loads TESS SSC lightcurve centroid data
    """
    flag = ''
    
    hdu = fits.open(filepath)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']

    if tc_per is not None:
        if tc_per == 0:
            tc_per = time[-1] - time[0]
        if tc_tdur / tc_per > 0.2:
            tc_tdur = tc_per*0.2
    X = hdu[1].data['MOM_CENTR1']
    Y = hdu[1].data['MOM_CENTR2']
    cam = hdu[0].header['CAMERA']
    ccd = hdu[0].header['CCD']   
    
    nancut = np.isnan(time) | np.isnan(X) | np.isnan(Y) | np.isnan(flux)
    time = time[~nancut]
    X = X[~nancut]
    Y = Y[~nancut]
    
    if len(time) <= len(hdu[1].data['TIME'])/2 or len(hdu[1].data['TIME']) < 3:  # if nancut removed > half the points, or there weren't any anyway
        flag = 'Not enough data'
    

    hdu.close()
    del hdu
    
    if flag:
        return time, X, Y, flag, cam, ccd
    

    if tc_per is not None and tc_t0 is not None:
        normphase = np.abs((np.mod(time-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
        intransit = normphase <= 1
        
        if sum(intransit) == 0 and not flag:
            flag = 'No transit data' 
        elif sum(intransit) <= 2 and not flag:
            flag = 'Not enough transit data'

    if trim: 
        exp_time = np.nanmedian(np.diff(time)) * 24 * 60
        exp_time = int(np.round(exp_time))       
        if exp_time <= 2.1:
            cut_num = 360
        elif exp_time <= 11:
            cut_num = 72
        else:
            cut_num = 24
            
        # Remove the first 12 hours of observations
        time = time[cut_num:len(time)]
        X = X[cut_num:len(X)]
        Y = Y[cut_num:len(Y)]
        
        if tc_per is not None and tc_t0 is not None:
            normphase = np.abs((np.mod(time-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
            intransit = normphase <= 1
            
            if sum(intransit) == 0 and not flag:
                flag = 'No transit data after trim' 
            elif sum(intransit) <= 2 and not flag:
                flag = 'Not enough transit data after trim'
            
            
 
    if flatten:
        Xcurve = np.array([time, X, np.ones(len(time))]).T
        Ycurve = np.array([time, Y, np.ones(len(time))]).T

        X = tsf.TESSflatten(Xcurve, split=True, winsize=2, stepsize=0.15, polydeg=3,
                            niter=10, sigmaclip=4., gapthresh=100., transitcut=transitcut,
                            tc_per=tc_per, tc_t0=tc_t0, tc_tdur=tc_tdur, divide=False, centroid=False)
        Y = tsf.TESSflatten(Ycurve, split=True, winsize=2, stepsize=0.15, polydeg=3,
                            niter=10, sigmaclip=4., gapthresh=100., transitcut=transitcut,
                            tc_per=tc_per, tc_t0=tc_t0, tc_tdur=tc_tdur, divide=False, centroid=False)
        
    if cut_outliers:
        if tc_per is not None and tc_t0 is not None:
            MAD_X = futils.MAD(X[~intransit])
            MAD_Y = futils.MAD(Y[~intransit])
        else:
            MAD_X = futils.MAD(X)
            MAD_Y = futils.MAD(Y)
            
        if (MAD_X == 0 or MAD_Y == 0) and not flag:
            flag = 'MAD is 0'

        cut = (np.abs(X-np.median(X))/MAD_X < cut_outliers) & (np.abs(Y-np.median(Y))/MAD_Y < cut_outliers)

        # avoid removing too much (happens with discontinuities for example)
        while np.sum(cut) < 3*len(X)/4:
            cut_outliers = cut_outliers + 1
            cut = (np.abs(X-np.median(X))/MAD_X < cut_outliers) & (np.abs(Y-np.median(Y))/MAD_Y < cut_outliers)
        
        if tc_per is not None and tc_t0 is not None:  
            cut[intransit] = True  # never cut points in the transit, too much risk they'll be marked as outliers
        
        time = time[cut]
        X = X[cut]
        Y = Y[cut]
        
        if tc_per is not None and tc_t0 is not None:
            normphase = np.abs((np.mod(time-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
            intransit = normphase <= 1
            
            mad_x_in = futils.MAD(X[intransit])
            mad_y_in = futils.MAD(Y[intransit])
            index_in = np.arange(len(time))[intransit]
            
            cut = (np.abs(X[intransit]-np.median(X[intransit]))/mad_x_in > 6) | (np.abs(Y[intransit]-np.median(Y[intransit]))/mad_y_in > 6)
            index_cut = index_in[cut]
            
            X = np.delete(X, index_cut) 
            Y = np.delete(Y, index_cut) 
            time = np.delete(time, index_cut) 
                 
    return time, X, Y, flag, cam, ccd


def load_spoc_masks(filepath, background=False):
    hdu = fits.open(filepath)
    wcs = WCS(hdu[2].header)
    mask_data = hdu[2].data
    cam = hdu[0].header['CAMERA']
    ccd = hdu[0].header['CCD']
    
    origin = hdu[2].header['CRVAL1P'], hdu[2].header['CRVAL2P']
    
    # Use the bit information to retrieve the aperture or centroid masks. 
    # The centroid mask will be the same as the aperture for the majority of cases.
    aperture = np.bitwise_and(mask_data, 2) / 2
    centroid = np.bitwise_and(mask_data, 8) / 8
    
    if background:
        background = np.bitwise_and(mask_data, 4) / 4
        hdu.close()
        del hdu
        return aperture, centroid, wcs, origin, cam, ccd, background
    else:
        hdu.close()
        del hdu
        return aperture, centroid, wcs, origin, cam, ccd
                
                                    
def centroid_fitting(ticid, candidate, sector, time, X, Y, per, t0, tdur, tdur23, loss='linear', plot=False): 
    if per == 0:
        per = time[-1] - time[0]    
    normphase = np.abs((np.mod(time-t0-per*0.5, per) - 0.5*per) / (0.5*tdur))
    
    intransit = normphase <= 1  
    intransit_half = normphase <= 0.5 # avoids half of transit to minimise ingress affecting result
    nearby = (normphase > 1) & (normphase < 3)
    
    if sum(intransit_half) == 0:
        flag = 'No half transit points'
    elif sum(intransit_half) < 3:
        flag = 'Not enough half transit points'
    elif sum(nearby) < 6:
        flag = 'No nearby points'
    else:
        flag = ''
    
    if not flag:            
        from scipy import optimize
        def _Trapezoidmodel(phase_data, t23, t14, depth):
            t0_phase = 0.5
            centrediffs = np.abs(phase_data - t0_phase)
            model = np.zeros_like(phase_data)
            model[centrediffs<t23/2.] = depth
            in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
            model[in_gress] = depth + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*(-depth)
            if t23>t14:
                model = np.ones(len(phase_data))*1e8
            return model   

        phase = futils.phasefold(time,per, t0+per*0.5)  #transit at phase 0.5
        idx = np.argsort(phase)
        phase = phase[idx]
               
        initialguess = [tdur23 / per, tdur / per, 0]
        bounds=[(initialguess[0]*0.95, initialguess[1]*0.99, -np.inf),(initialguess[0]*1.05, initialguess[1]*1.01, np.inf)]
        
        X_scatter = futils.MAD(X[~intransit])
        Y_scatter = futils.MAD(Y[~intransit])
                
        try:
            xfit = optimize.curve_fit(_Trapezoidmodel, phase, X[idx], 
                                    p0=initialguess, sigma=np.full_like(X, X_scatter),
                                    bounds=bounds,
                                    absolute_sigma=False, loss=loss)
                            
            yfit = optimize.curve_fit(_Trapezoidmodel, phase, Y[idx], 
                                    p0=initialguess, sigma=np.full_like(Y, Y_scatter),
                                    bounds=bounds,
                                    absolute_sigma=False, loss=loss)       
        
            x_diff = xfit[0][2]
            x_err = np.sqrt(np.diag(xfit[1]))[2] 
            y_diff = yfit[0][2]
            y_err = np.sqrt(np.diag(yfit[1]))[2]
            
            if x_err >= 1 or y_err >= 1:
                x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
                flag = 'Fit did not converge'
                
            if x_diff < X_scatter and y_diff < Y_scatter and (X_scatter > 0.002 or Y_scatter > 0.002):
                x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
                flag = 'Too much scatter in centroid data'
                               
            if plot:
                modelX = _Trapezoidmodel(phase, *xfit[0])
                modelY = _Trapezoidmodel(phase, *yfit[0])

                phase -= 0.5
                
                limit = np.abs(phase) < tdur*10/per
                
                fig, ax = plt.subplots(1,1, figsize=(8,8))
                fig.suptitle(f'TIC {ticid} - {candidate} Sector {sector}')
                ax.scatter(phase[limit], X[idx][limit], s=0.5, c='k')
                ax.plot(phase[limit], modelX[limit], c='darkorange', lw=2)
                ax.scatter(phase[limit], Y[idx][limit] - 0.01, s=0.5, c='k')
                ax.plot(phase[limit], modelY[limit]-0.01, c='darkorange', lw=2)
                ax.set_xlabel('Phase', fontsize=14)
                ax.set_ylabel('Normalized Centroid Position', fontsize=14)
                
                outfile = Path(__file__).resolve().parents[1] / 'Output' / 'Plots' / f'{ticid}' 
                outfile.mkdir(exist_ok=True)
                outfile = outfile / f'centroidfit_{ticid}_{candidate}_{sector}.png'
                fig.savefig(outfile, bbox_inches='tight')
                 
        except Exception:
            x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
            flag = 'Fit fail'
    else:
        x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
    
    return x_diff, x_err, y_diff, y_err, flag


def nearby_depth(depth, f_t, f_n):
    depth_n = depth * np.divide(f_t, f_n, out=np.zeros_like(f_n), where=f_n!=0.0)

    return depth_n


def test_target_aperture(x, y, aperture):
    xint = int(np.floor(x+0.5))
    yint = int(np.floor(y+0.5))
        
    if xint < 0 or xint >= aperture.shape[1] or yint < 0 or yint >= aperture.shape[0]:
        test = False
    else:
        test = bool(aperture[yint, xint])
        
    return test
    


def sources_prf(sec, cam, ccd, origin, X_sources, Y_sources, shape):
    '''
    Returns the prf models centred on each source for all pixels of the target pixel
    '''
    from CandidateSet import PRF
    
    prfs = np.zeros([X_sources.size, shape[0], shape[1]])
    
    tp_x = np.round(origin[0] + X_sources[0])
    tp_y = np.round(origin[1] + Y_sources[0])

    if sec < 4:
        prf_folder = Path.cwd() / 'CandidateSet' / 'PRF files' / 'Sector 1'
    else:
        prf_folder = Path.cwd() / 'CandidateSet' / 'PRF files' / 'Sector 4'
                
    prf_model = PRF.TESS_PRF(cam, ccd, sec, tp_x, tp_y, prf_folder)
    
    for i in range(len(X_sources)):
        prfs[i] = prf_model.locate(X_sources[i], Y_sources[i], shape)
        
    return prfs
    
    
def flux_fraction_in_ap(fluxes, aperture, source_prfs):     
    aperture_flux = source_prfs*fluxes[:, None, None] #convert fractions to actual fluxes within each pixel (still by star)    
    
    aperture_flux *= aperture # Multiplies by either 1 or 0 if pixel is in the aperture mask or not
    total_ap_flux = np.sum(aperture_flux)
    
    flux_fractions = []
    
    for i in range(len(fluxes)):
        source_ap_flux = np.sum(aperture_flux[i])
        flux_fractions.append(source_ap_flux/total_ap_flux)
        
    return flux_fractions, total_ap_flux  


def model_centroid(aperture, fluxes, fluxfractions):
    # Identify the pixels used in the aperture
    pixels = np.where(aperture == 1)

    # Extract a 2x2 array of the mask grid. Pixels not used will retain 0 in their values.
    y_min = np.min(pixels[0])
    y_max = np.max(pixels[0]) + 1
    
    x_min = np.min(pixels[1])
    x_max = np.max(pixels[1]) + 1
    
    aperture_only = aperture[y_min:y_max, x_min:x_max]
    fluxfractions = fluxfractions * fluxes[:, None, None] #convert fractions to actual fluxes within each pixel (still by star)
    fluxfractions = np.sum(fluxfractions, axis=0) #should now be 2D.
    
    fluxfractions = fluxfractions[y_min:y_max, x_min:x_max]
 
    fluxfractions *= aperture_only # Multiplies by either 1 or 0 if pixel is in the aperture mask or not
    #need to specify indices in below two lines. Depends on format of aperture.
    X = np.average(np.arange(x_min, x_max),weights=np.sum(fluxfractions,axis=0)) #sums fluxfractions across y axis to leave x behind
    Y = np.average(np.arange(y_min, y_max),weights=np.sum(fluxfractions,axis=1)) #sums fluxfractions across x axis to leave y behind
    
    return X, Y


def calc_centroid_probability(cent_x, cent_y, cent_x_err, cent_y_err, diff_x, diff_y, diff_x_err, diff_y_err):
    X_err = np.sqrt(cent_x_err ** 2 + diff_x_err ** 2)
    Y_err = np.sqrt(cent_y_err ** 2 + diff_y_err ** 2)
    
    if np.isnan(X_err) or np.isnan(Y_err):
        return np.nan
    
    from scipy import spatial
    # get mahalanobis distance
    cov = np.array([[X_err ** 2, 0], [0, Y_err ** 2]])
    VI = np.linalg.inv(cov)
    mahalanobis = spatial.distance.mahalanobis([diff_x, diff_y], [cent_x, cent_y], VI)

    prob_centroid = np.exp(-(mahalanobis ** 2) / 2)
    
    return prob_centroid


def plot_lc(lc, ticid, candidate=None, phasefold=False, per=None, t0=None, tdur=None):
    fig, (ax1) = plt.subplots(1,1, figsize=(12,6))
    
    if candidate:
        fig.suptitle(f'{ticid}-{candidate}')
    else:
        fig.suptitle(f'{ticid}')
    
    if phasefold:
        if per is None or t0 is None:
            print('Need period and epoch to phase fold')
            return
        
        phase = futils.phasefold(lc['time'], per, t0 - per*0.5) - 0.5
        idx = np.argsort(phase)
        phase = phase[idx]
        flux = lc['flux'][idx]
        
        ax1.scatter(phase, flux, s=0.5)
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Normalised Flux')
        
        if tdur is not None:
            intransit = np.where(np.abs(phase)<=0.5*(tdur/per))[0]
            ax1.scatter(phase[intransit], flux[intransit], s=0.5, c='r')
    else:
        ax1.scatter(lc['time'], lc['flux'], s=0.5)
        ax1.set_xlabel('Time [BJD]')
        ax1.set_ylabel('Normalised Flux')
           