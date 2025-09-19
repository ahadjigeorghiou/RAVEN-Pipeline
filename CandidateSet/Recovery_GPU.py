from pathlib import Path
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from astropy.io import fits
from TrainingSet import utils
from CandidateSet import utils as cutils
from Features import utils as futils
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from TrainingSet import gpu_bls as gbls


def BLS_gpu(time, flux, min_period, max_period, functions=None, save=False, output=None):
    # set up search parameters
    search_params = dict(
                        # Searches q values in the range
                        # (q0 * qmin_fac, q0 * qmax_fac)
                        # where q0 = q0(f, rho) is the fiducial
                        # q value for Keplerian transit around
                        # star with mean density rho
                        qmin_fac=0.5,
                        qmax_fac=2.0,

                        # Assumed mean stellar density
                        rho=1.0,

                        # The min/max frequencies as a fraction
                        # of their autoset values
                        fmin_fac=1.0,
                        fmax_fac=1.0,

                        # oversampling factor; frequency spacing
                        # is multiplied by 1/samples_per_peak
                        samples_per_peak=10,

                        # The logarithmic spacing of q
                        dlogq=0.1,

                        # Number of overlapping phase bins
                        # to use for finding the best phi0
                        noverlap=3,
                        
                        fmin=1/max_period, 
                        fmax=1/min_period, 
                        use_fast=True,
                        functions=functions)
    
    dflux = np.full_like(flux, futils.MAD(flux))
    freqs, power, qmins, qmaxes = gbls.eebls_transit_gpu(time, flux, dflux,
                                          **search_params)
    
    periods = 1/freqs
    idx = np.argsort(periods)
    periods = periods[idx]
    
    power = power[idx]
    qmins = qmins[idx]
    qmaxes = qmaxes[idx]
    
    if save:
        hdu = fits.BinTableHDU.from_columns([fits.Column(name='Period', format='E', array=periods),
                                             fits.Column(name='Power', format='E', array=power),
                                             fits.Column(name='qmins', format='E', array=qmins),
                                             fits.Column(name='qmaxes', format='E', array=qmaxes)])
        hdu.writeto(output)
                    
    return periods, power, qmins, qmaxes


def gpu_bls_results(time, flux, periods, qmins, qmaxes, sde, sde_limit, functions):
    results = []
    
    peaks = utils.find_BLS_peaks(sde, sde_limit, periods)
    
    peak_pers = periods[peaks]
    peak_sdes = sde[peaks]
    best_f = 1/peak_pers
    dflux = np.full_like(flux, futils.MAD(flux))
    sols = gbls.eebls_gpu(time, flux, dflux, best_f, 
                            qmin=qmins[peaks], qmax=qmaxes[peaks], dlogq=0.1, 
                            noverlap=3, functions=functions,
                            ignore_negative_delta_sols=True, freq_batch_size=len(peaks))[1]
    
    for i, sol in enumerate(sols):
        peak_sde = peak_sdes[i]
        peak_per = peak_pers[i]
        
        peak_tdur = sol[0]*peak_pers[i]
        
        peak_t0 = utils.get_bls_t0(time, best_f[i], sol[1], sol[0])
        
        peak_depth = futils.calculate_depth(time, flux, peak_per, peak_t0, peak_tdur)[0]*1e6
        
        mes, phase = futils.MES(flux, time, peak_per, peak_t0, peak_tdur, peak_depth*1e-6)[:2]
        
        peak_mes = mes[futils.find_index(phase, 0)]
        
        tdur_p = peak_tdur/peak_per
        
        intransit = np.abs(phase) <= tdur_p * 0.5    
        idx = np.argmax(mes[intransit])
        phase_max_int = phase[intransit][idx]
        mes_max_int = mes[intransit][idx]
        
        idx = np.argmax(mes)
        phase_max = phase[idx]
        mes_max = mes[idx]
        
        results.append((i+1, peak_sde, peak_per, peak_t0, peak_tdur, peak_depth, peak_mes, mes_max_int, phase_max_int, mes_max, phase_max))
            
    return results


def single_per_bls(time, flux, period):
    duration = np.arange(1e-3,0.351, 1e-3)*period
    duration = duration[(duration >= 0.0625) & (duration <= 2.0)]
    
    results = BoxLeastSquares(time, flux).power([period], duration, oversample=3, objective='snr')
    
    return results['transit_time'][0], results['duration'][0], results['depth'][0]*1e6


def read_bls_output(f):
    hdu = fits.open(f)
    
    periods = hdu[1].data['Period']
    power = hdu[1].data['Power']
    qmins = hdu[1].data['qmins']
    qmaxes = hdu[1].data['qmaxes']
    
    hdu.close()
    
    return periods, power, qmins, qmaxes
   
    
class BLSRecovery(object):
    def __init__(self, infile, infile_type='archive', sector_file='SPOC_sectors.csv', output='default', lc_location='default', dir_style='per_target', sde_limit=7, min_period = 0.5, max_period = 16, 
                 sector_limit=[], multiprocessing=0, save_suffix=None, load_suffix=None):
        raven_dir = Path(__file__).resolve().parents[1]  
        
        if output == 'default':
            self.output = raven_dir / 'Output' 
        else:
            self.output = Path(output)
        
        if infile_type == 'exofop':
            self.data = cutils.load_exofop_toi(infile)
        elif infile_type == 'archive':
            self.data = cutils.load_archive_toi(infile)
        elif infile_type == 'default':
            self.data = cutils.load_default(infile)
        elif infile_type == 'array' or infile_type == 'list':
            if Path(infile).suffix == '.txt' or Path(infile).suffix == '.csv':
                ticids = np.genfromtxt(raven_dir / 'Input' / infile, dtype=np.int64)
            elif Path(infile).suffix == '.npy':
                ticids = np.load(raven_dir / 'Input' / infile)
            self.data = pd.DataFrame(index=pd.Index(ticids, name='ticid'))
        elif infile_type == 'sectors':
            self.data = None
        else:
            raise ValueError('Infile type not supported.')
        
        sectors =  pd.read_csv(raven_dir / f'Input/{sector_file}').set_index('ticid')
        
        if self.data is not None:
            self.data = self.data.join(sectors, how='inner')
        else:
            self.data = sectors
        
        self.data.rename(columns={'sector':'sectors'}, inplace=True)
        
        if lc_location == 'default':
            self.lc_location = raven_dir / 'Lightcurves'
        else:
            self.lc_location = Path(lc_location)
        
        self.dir_style = dir_style
        self.sde_limit = sde_limit
        self.min_period = min_period
        self.max_period = max_period
        self.sector_limit = sector_limit
        self.multiprocessing = multiprocessing
        
        if sector_limit:
            self.data['sectors'] = self.data['sectors'].apply(lambda x: [int(s) for s in x.split(',') if (len(s) > 0 and (int(s) >= sector_limit[0] and int(s) <= sector_limit[1]))])
        else:
            self.data['sectors'] = self.data['sectors'].apply(lambda x: [int(s) for s in x.split(',') if len(s)>0])

        
        # self.data['sec_num'] = [len(s) for s in self.data['sectors']]
        # self.data = self.data.query('sec_num > 0') 
               
        if save_suffix is None:
            if sector_limit:
                self.save_suffix = f'{sector_limit[0]}-{sector_limit[1]}'
            else:
                self.save_suffix = ''
        else:
            self.save_suffix = save_suffix
        
        if load_suffix is not None:
            infile = self.output / f'Recovery_{load_suffix}.csv'
            try:
                self.recovered = pd.read_csv(infile).set_index(['ticid', 'peak_sig'])
            except FileNotFoundError:
                print(f'File: {infile} not found.')
                self.recovered = pd.DataFrame()
        else:
            self.recovered = pd.DataFrame()
            
        self.recovery_results = pd.DataFrame()


    def dataset_bls(self, bls_output_loc='default', save_bls_output=False, reuse_output=False, reuse_peaks=False):
        if bls_output_loc == 'default':
            self.bls_output = self.output / 'BLS Output' 
        else:
            self.bls_output = Path(bls_output_loc)
        
        if save_bls_output:
            self.bls_output.mkdir(exist_ok = True)
            
        self.save_bls_output = save_bls_output    
        self.reuse_output = reuse_output
        self.reuse_peaks = reuse_peaks
            
        if len(self.recovered) > 0:
            tics_to_run = self.data.index.unique('ticid').difference(self.recovered.index.unique('ticid'))
        else:
            tics_to_run = self.data.index.unique('ticid')
            
        tic_num = len(tics_to_run)

        if tic_num > 0:
            df_lst = []
            print(f'Running BLS recovery for {tic_num} targets')
            if self.multiprocessing > 1 and tic_num > 10:
                if tic_num < self.multiprocessing:
                    workers = tic_num
                else:
                    workers = self.multiprocessing
                print(f'Running multiprocessing on {workers} processors')
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
                    try:
                        factor = 15
                        while tic_num < factor*workers:
                            factor -= 1
                            if factor == 1:
                                break
                        tic_split = np.array_split(tics_to_run, factor*workers)
                        futures = {ex.submit(self.multi_bls, tics): tics for tics in tic_split}
                        
                        for future in as_completed(futures):
                            try:
                                df, fails = future.result()
                                df_lst.append(df)
                                
                                if len(fails) > 0:
                                    for fail in fails:
                                        print(f'Exception {fail[1]} occured for ticid: {fail[0]}')
                            except Exception as e:
                                group = futures[future]
                                print(f'Exception {e} occured for tic group: {group}')
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        # Attempt to save
                        recovery_results = pd.concat(df_lst)
                        if len(self.recovered) > 0:
                            self.recovered = pd.concat([self.recovered, recovery_results])
                        else:
                            self.recovered = recovery_results
                        
                        if len(self.recovered) > 0:
                            self.recovered.sort_values(['ticid', 'peak_sig'], inplace=True)
                            self.recovered.to_csv(self.output/f'Recovery_{self.save_suffix}.csv')
                        raise ValueError('Keyboard interrupt')            
            else:
                functions = gbls.compile_bls()
                for ticid in tics_to_run:
                    try:
                        df = self.run_bls(self.data.loc[[ticid]], functions)
                        df_lst.append(df)
                    except Exception as e:
                        print(f'Exception {e} occured for ticid: {ticid}')
                        
            recovery_results = pd.concat(df_lst)
            # Save results
            if len(self.recovered) > 0:
                self.recovered = pd.concat([self.recovered, recovery_results])
            else:
                self.recovered = recovery_results
            
            if len(self.recovered) > 0:
                self.recovered.sort_values(['ticid', 'peak_sig'], inplace=True)
                self.recovered.to_csv(self.output/f'Recovery_{self.save_suffix}.csv')
            else:
                print('No recovery results')
    
    
    def multi_bls(self, ticids, functions=None):
        df_lst = [pd.DataFrame()]
        fails = []
        if functions is None:
            functions = gbls.compile_bls()
        for tic in ticids:
            try:
                df = self.run_bls(self.data.loc[[tic]], functions)
                df_lst.append(df)
            except Exception as e:
                fails.append((tic, e))
                
        df_multi = pd.concat(df_lst)
        
        return df_multi, fails
        
        
    def run_bls(self, data, functions):
        
        ticid = data.index.unique('ticid').values[0]
        
        if self.reuse_output:
            infile = self.bls_output / f'{ticid}_bls.fits'
            try:
                periods, power, qmins, qmaxes = read_bls_output(infile)
            except FileNotFoundError:
                periods, power, qmins, qmaxes = None, None, None, None  
        else:
            periods, power, qmins, qmaxes = None, None, None, None  
        
        if periods is None:
            # Empty arrays to stitch the sector lcs into one
            fluxes = []
            times = []
            
            # Load the detrended and normalised lightcurves per sector of observation
            for sec in data.iloc[0]['sectors']:
                lcfile = cutils.lc_filepath(self.lc_location, self.dir_style, ticid, sec)
                
                try:            
                    lc = cutils.load_spoc_lc(lcfile, flatten=True, transitcut=False, winsize=4.0)
                except FileNotFoundError:
                    continue
                
                fluxes.append(lc['flux'])
                times.append(lc['time'])
                
            if len(fluxes) > 0:
                flux = np.concatenate(fluxes)
                time = np.concatenate(times)
            else:
                df = pd.DataFrame(0, index=[0] ,columns=['peak_sig', 'peak_sde', 'peak_per', 'peak_t0', 'peak_tdur', 'peak_depth', 'peak_mes', 'peak_mes_int', 'peak_phase_int', 'max_mes', 'max_phase'])
                df['ticid'] = ticid

                df.set_index(['ticid', 'peak_sig'], inplace=True)
                
                return df

            lc = {'flux':flux,
                  'time':time}
            
            if self.save_bls_output:
                outfile = self.bls_output / f'{ticid}_bls.fits'
            else:
                outfile = None
            periods, power, qmins, qmaxes = BLS_gpu(lc['time'], lc['flux'], self.min_period, self.max_period, functions=functions, save=self.save_bls_output, output=outfile)
        
        sde = utils.get_BLS_sde(periods, power, int(len(power)/15))
        results = gpu_bls_results(lc['time'], lc['flux'], periods, qmins, qmaxes, sde, self.sde_limit, functions=functions)
            
        df = pd.DataFrame(results, columns=['peak_sig', 'peak_sde', 'peak_per', 'peak_t0', 'peak_tdur', 'peak_depth', 'peak_mes', 'peak_mes_int', 'peak_phase_int', 'max_mes', 'max_phase'])
        df['ticid'] = ticid

        df.set_index(['ticid', 'peak_sig'], inplace=True)

        return df
    
    
    def dataset_recovery(self):
        if len(self.recovered) == 0:
            raise ValueError('Run BLS first!')
        
        if 'candidate' not in self.data.index.names:
            raise ValueError('No candidates found to compare to.')
         
        self.recovered = self.recovered[['peak_sde', 'peak_per', 'peak_t0', 'peak_tdur',
                                         'peak_depth', 'peak_mes', 'peak_mes_int',
                                         'peak_phase_int', 'max_mes', 'max_phase']]
        
        data_copy = self.data.copy()
        data_copy.reset_index(inplace=True)
        data_copy.set_index('ticid', inplace=True)
        
        df = self.recovered.copy()
        df.reset_index(inplace=True)
        df.set_index('ticid', inplace=True)
        
        df = df.join(data_copy[['candidate', 't0', 'per', 'depth']], on='ticid', how='inner')
        
        df['p_ratio'] = df['peak_per']/df['per']
        df['p_ratio_inv'] = df['per']/df['peak_per']
        df['depth_ratio'] = df['peak_depth']/df['depth']
        df['depth_ratio_inv'] = df['depth']/df['peak_depth']
        df['t0_offset1'] = np.round((df['t0'] - df['peak_t0'])/df['peak_per'])
        df['peak_t0_adj1'] = (df['peak_t0'] + df['peak_per']*df['t0_offset1'])
        df['t0_diff1'] = np.abs(df['t0'] - df['peak_t0_adj1'])
        df['t0_offset2'] = np.round((df['t0'] - df['peak_t0'])/df['per'])
        df['peak_t0_adj2'] = (df['peak_t0'] + df['per']*df['t0_offset2'])
        df['t0_diff2'] = np.abs(df['t0'] - df['peak_t0_adj2'])
        
        df.set_index([df.index, 'candidate', 'peak_sig'], inplace=True)
        
        t0_cond = '(t0_diff1 <= 0.5 or t0_diff2 <= 0.5)'
        depth_cond = 'depth_ratio < 3 and depth_ratio_inv < 3'

        df['Recovered'] = 'False' # Set all to false
        df.loc[df.query('p_ratio >= 0.99 and p_ratio <= 1.01').index, 'Recovered'] = 'Pmatch'
        df.loc[df.query(f'p_ratio >= 0.98 and p_ratio <= 1.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'True'
        df.loc[df.query(f'p_ratio >= 1.98 and p_ratio <= 2.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx2'
        df.loc[df.query(f'p_ratio >= 2.98 and p_ratio <= 3.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx3'
        df.loc[df.query(f'p_ratio_inv >= 1.98 and p_ratio_inv <= 2.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx05'
        df.loc[df.query(f'p_ratio_inv >= 2.98 and p_ratio_inv <= 3.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx03'
        
        df.loc[df.query('peak_sig == 0').index, 'Recovered'] = 'False'
        df.loc[df.query('peak_depth < 0').index, 'Recovered'] = 'False'
        
        df.drop(['t0', 'per', 'depth_ratio', 'depth_ratio_inv', 't0_offset1', 't0_offset2', 'peak_t0_adj1', 'peak_t0_adj2', 't0_diff1', 't0_diff2'], axis=1, inplace=True)
        
        # df.reset_index(inplace=True)
        # df.set_index(['ticid', 'candidate'], inplace=True)
        df.sort_values(['ticid', 'candidate', 'peak_sig'], inplace=True)
        
        self.recovery_results = df

        self.recovery_results.to_csv(self.output/f'Recovery_{self.save_suffix}_All.csv')
    
    def peak_mes_recovery(self, mes_threshold=0.8):
        if 'Recovered' not in self.recovery_results.columns:
            raise KeyError('Run dataset_recovery first!')
        
        self.recovery_results['peak_mes'] = self.recovery_results['peak_mes'].fillna(0)
        
        # self.recovery_results.set_index([self.recovery_results.index, 'peak_sig'], inplace=True)
        
        self.recovery_results.loc[self.recovery_results.query(f'peak_mes < {mes_threshold}').index, 'Recovered'] = 'False'
        
        # self.recovery_results.reset_index(inplace=True)
        
        # self.recovery_results.set_index(['ticid', 'candidate'], inplace=True)
            

    def reduced_recovery(self):
        if len(self.recovered) == 0:
            raise ValueError('Run dataset_bls first!')
        
        if 'Recovered' not in self.recovery_results.columns:
            raise KeyError('Run dataset_recovery first!')
        
        df = self.recovery_results.copy()
        
        df.reset_index(inplace=True)
        df.set_index(['ticid','candidate'], inplace=True)
        
        df.query('peak_sig != 10', inplace=True)
               
        fail_df = df.query('peak_sig == 0')
        
        df.drop(fail_df.index, inplace=True)
        
        true_df = df.query('Recovered == "True"').groupby(['ticid', 'candidate']).first()
        
        df.drop(true_df.index, inplace=True)
          
        alias_df = df[df['Recovered'].isin(['Aliasx2', 'Aliasx3', 'Aliasx03', 'Aliasx05'])]
        alias_df = alias_df.groupby(['ticid', 'candidate']).first()
        
        df.drop(alias_df.index, inplace=True)
        
        pmatch_df = df.query('Recovered == "Pmatch"').groupby(['ticid', 'candidate']).first()
        
        df.drop(pmatch_df.index, inplace=True)
        
        false_df = df.groupby(['ticid', 'candidate']).first()
        
        self.recovery_results = pd.concat([fail_df, true_df, alias_df, pmatch_df, false_df]).sort_values('candidate')

        self.recovery_results.to_csv(self.output/f'Recovery_{self.save_suffix}_Redux.csv')
        
