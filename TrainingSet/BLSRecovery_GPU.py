from pathlib import Path
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from astropy.io import fits
from TrainingSet import utils
from Features import utils as futils
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from TrainingSet import gpu_bls_new as gbls


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
    peaks = utils.find_BLS_peaks(sde, sde_limit, periods)
    results = []
    if len(peaks) == 0:
        peak = 0
        peak_sde = np.max(sde)
        idx = np.argmax(sde)
        
        peak_per = periods[idx]
        
        dflux = np.full_like(flux, futils.MAD(flux))
        
        sol = gbls.eebls_gpu(time, flux, dflux, 1/np.array([peak_per]), 
                        qmin=qmins[idx], qmax=qmaxes[idx], dlogq=0.1, 
                        noverlap=3, functions=functions,
                        ignore_negative_delta_sols=True, freq_batch_size=1)[1][0]
        
        peak_tdur = sol[0]*peak_per
            
        peak_t0 = utils.get_bls_t0(time, 1/peak_per, sol[1], sol[0])
            
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
            
        results.append((peak, peak_sde, peak_per, peak_t0, peak_tdur, peak_depth, peak_mes, mes_max_int, phase_max_int, mes_max, phase_max))
    else:
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


def read_bls_output(f):
    hdu = fits.open(f)
    
    periods = hdu[1].data['Period']
    power = hdu[1].data['Power']
    qmins = hdu[1].data['qmins']
    qmaxes = hdu[1].data['qmaxes']
    
    hdu.close()
    
    return periods, power, qmins, qmaxes
   
    
class BLSRecovery(object):
    def __init__(self, sim_type, output='default', lc_location='default', 
                 sde_limit=5, min_period = 0.5, max_period = 16, sector_limit=[], 
                 multiprocessing=0, save_suffix=None, load_suffix=None):
        
        self.sim_type = sim_type
        
        if output == 'default':
            self.output = Path(__file__).resolve().parents[0] / 'Output' 
        else:
            self.output = Path(output)
        
        try:
            self.injection_log = pd.read_csv(self.output / f'{sim_type}_InjectionLog.csv').set_index(['sim_batch', 'sim_num'])
        except FileNotFoundError:
            raise ValueError(f'Injection log for the synthetic lc set not found at location: {self.output}')
        
        self.injection_log.loc[self.injection_log['sectors'].isna(), 'sectors'] = ''
        
        if lc_location == 'default':
            self.lc_location = Path(__file__).resolve().parents[0] / 'InjectedSet' / f'{sim_type}'
        else:
            self.lc_location = Path(lc_location)
        
        self.sde_limit = sde_limit
        self.min_period = min_period
        self.max_period = max_period
        self.sector_limit = sector_limit
        if multiprocessing == 1:
            multiprocessing = os.cpu_count()
        self.multiprocessing = multiprocessing
        
        if sector_limit:
            self.injection_log['sectors'] = self.injection_log['sectors'].apply(lambda x: [int(s) for s in x.split(',') if (len(s) > 0 and (int(s) >= sector_limit[0] and int(s) <= sector_limit[1]))])
        else:
            self.injection_log['sectors'] = self.injection_log['sectors'].apply(lambda x: [int(s) for s in x.split(',') if len(s)>0])

        
        self.injection_log['sec_num'] = [len(s) for s in self.injection_log['sectors']]
        self.injection_log = self.injection_log.query('sec_num > 0') 
               
        if save_suffix is None:
            if sector_limit:
                self.save_suffix = f'{sector_limit[0]}-{sector_limit[1]}'
            else:
                self.save_suffix = ''
        else:
            self.save_suffix = save_suffix
        
        if load_suffix is not None:
            infile = self.output / f'{sim_type}_Recovery_{load_suffix}.csv'
            try:
                self.recovered = pd.read_csv(infile).set_index(['sim_batch', 'sim_num'])
            except FileNotFoundError:
                print(f'File: {infile} not found.')
                self.recovered = pd.DataFrame()
        else:
            self.recovered = pd.DataFrame()


    def dataset_bls(self, bls_output_loc='default', save_bls_output=False, reuse_output=False):
        if bls_output_loc == 'default':
            self.bls_output = self.output / 'BLS Output' 
        else:
            self.bls_output = Path(bls_output_loc)
        
        if save_bls_output:
            self.bls_output.mkdir(exist_ok = True)
            
        self.save_bls_output = save_bls_output    
        self.reuse_output = reuse_output
            
        if len(self.recovered) > 0:
            sims_to_run = self.injection_log.index.difference(self.recovered.index)
        else:
            sims_to_run = self.injection_log.index
            
        sim_num = len(sims_to_run)
        if sim_num > 0:
            print(f'Running BLS recovery for {sim_num} simulations')
            if self.multiprocessing > 1 and sim_num > 4:
                if sim_num < self.multiprocessing:
                    workers = sim_num
                else:
                    workers = self.multiprocessing
                print(f'Running multiprocessing on {workers} processors')
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
                    try:
                        factor = 15
                        while sim_num < 5*factor*workers:
                            factor -= 1
                            if factor == 1:
                                break
                        sim_split = np.array_split(sims_to_run, factor*workers)
                        futures = {ex.submit(self.multi_bls, sims): sims for sims in sim_split}
                        
                        for future in as_completed(futures):
                            try:
                                df, fails = future.result()
                                self.recovered = pd.concat([self.recovered, df])
                                
                                if len(fails) > 0:
                                    for fail in fails:
                                        print(f'Exception {fail[1]} occured for sim: {fail[0]}')
                            except Exception as e:
                                group = futures[future]
                                print(f'Exception {e} occured for sim group: {group}')
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        # Attempt to save
                        if len(self.recovered) > 0:
                            self.recovered.sort_values(['sim_batch', 'sim_num'], inplace=True)
                            self.recovered.to_csv(self.output/f'{self.sim_type}_Recovery_{self.save_suffix}.csv')
                        raise ValueError('Keyboard interrupt')            
            else:
                sim_split = np.array_split(sims_to_run, int(np.ceil(len(sims_to_run)/10)))
                functions = gbls.compile_bls()
                for sim_group in sim_split:
                    df, fails = self.multi_bls(sim_group, functions)
                    self.recovered = pd.concat([self.recovered, df])
                
                    for sim, e in fails:
                        print(f'Exception {e} occured for sim: {sim}')
            
            # Save results
            if len(self.recovered) > 0:
                self.recovered.sort_values(['sim_batch', 'sim_num'], inplace=True)
                self.recovered.to_csv(self.output/f'{self.sim_type}_Recovery_{self.save_suffix}.csv')
    
    
    def multi_bls(self, sims, functions=None):
        df_lst = [pd.DataFrame()]
        fails = []
        if functions is None:
            functions = gbls.compile_bls()
        for sim in sims:
            try:
                df = self.run_bls(self.injection_log.loc[[sim]], functions)
                df_lst.append(df)
            except Exception as e:
                fails.append((sim, e))
                
        df_multi = pd.concat(df_lst)
        
        return df_multi, fails
        
        
    def run_bls(self, data, functions):
        sim_type = self.sim_type
        
        sim_batch, sim_num = data.index[0]
        
        file = self.lc_location / f'{sim_type}-{sim_batch}-{sim_num}.fits'
        lc, sec_lcs = utils.load_injected_lc(file, self.sector_limit)

        sec_num = len(sec_lcs.keys())

        if self.save_bls_output:
            outfile = self.bls_output / f'{sim_type}-{sim_batch}-{sim_num}_bls.fits'
        else:
            outfile = None
        
        if self.reuse_output:
            infile = self.output / 'BLS Output'/ f'{sim_type}-{sim_batch}-{sim_num}_bls.fits'
            try:
                periods, power, qmins, qmaxes = read_bls_output(infile)
            except FileNotFoundError:
                periods, power, qmins, qmaxes = BLS_gpu(lc['time'], lc['flux'], self.min_period, self.max_period, functions=functions, save=self.save_bls_output, output=outfile)  
        else:   
            periods, power, qmins, qmaxes = BLS_gpu(lc['time'], lc['flux'], self.min_period, self.max_period, functions=functions, save=self.save_bls_output, output=outfile)  
        
        
        sde = utils.get_BLS_sde(periods, power, int(len(power)/15))
        results = gpu_bls_results(lc['time'], lc['flux'], periods, qmins, qmaxes, sde, self.sde_limit, functions=functions)
        
        df = pd.DataFrame(results, columns=['peak_sig', 'peak_sde', 'peak_per', 'peak_t0', 'peak_tdur', 'peak_depth', 'peak_mes', 'peak_mes_int', 'peak_phase_int', 'max_mes', 'max_phase'])
        df['sim_batch'] = sim_batch
        df['sim_num'] = sim_num
        df['sec_num'] = sec_num

        df.set_index(['sim_batch', 'sim_num'], inplace=True)

        return df
    
    
    def dataset_recovery(self):
        if len(self.recovered) == 0:
            raise ValueError('Run BLS first!')
        
        try:
            self.recovered.drop(['t0','per'], axis=1, inplace=True)
        except KeyError:
            pass
        
        self.recovered = self.recovered[['peak_sig', 'peak_sde', 'peak_per', 'peak_t0', 
                                         'peak_tdur','peak_depth', 'peak_mes', 'peak_mes_int',
                                         'peak_phase_int', 'max_mes', 'max_phase', 'sec_num']]
        
        df = self.recovered.join(self.injection_log[['t0', 'per', 'depth']], how='inner')
        
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
        
        df.set_index([df.index, 'peak_sig'], inplace=True)
        
        t0_cond = '(t0_diff1 <= 0.5 or t0_diff2 <= 0.5)'
        depth_cond = 'depth_ratio < 3 and depth_ratio_inv < 3'
        
        df['Recovered'] = 'False' # Set all to false       
        df.loc[df.query('p_ratio >= 0.99 and p_ratio <= 1.01').index, 'Recovered'] = 'Pmatch'
        df.loc[df.query(f'p_ratio >= 0.98 and p_ratio <= 1.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'True'
        df.loc[df.query(f'p_ratio >= 1.98 and p_ratio <= 2.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx2'
        df.loc[df.query(f'p_ratio >= 2.98 and p_ratio <= 3.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx3'
        df.loc[df.query(f'p_ratio_inv >= 1.98 and p_ratio_inv <= 2.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx05'
        df.loc[df.query(f'p_ratio_inv >= 2.98 and p_ratio_inv <= 3.02 and {t0_cond} and {depth_cond}').index, 'Recovered'] = 'Aliasx03'
        
        if 'sec_t0' in self.injection_log.columns:
            df = df.join(self.injection_log[['sec_t0', 'sec_depth']], how='inner')
            
            df['Secondary'] = False
            df['depth_ratio'] = df['peak_depth']/df['sec_depth']
            df['depth_ratio_inv'] = df['sec_depth']/df['peak_depth']
            df['t0_offset1'] = np.round((df['sec_t0'] - df['peak_t0'])/df['peak_per'])
            df['peak_t0_adj1'] = (df['peak_t0'] + df['peak_per']*df['t0_offset1'])
            df['t0_diff1'] = np.abs(df['sec_t0'] - df['peak_t0_adj1'])
            df['t0_offset2'] = np.round((df['sec_t0'] - df['peak_t0'])/df['per'])
            df['peak_t0_adj2'] = (df['peak_t0'] + df['per']*df['t0_offset2'])
            df['t0_diff2'] = np.abs(df['sec_t0'] - df['peak_t0_adj2'])
            
            df.loc[df.query(f'Recovered != "True" and p_ratio >= 0.98 and p_ratio <= 1.02 and {t0_cond} and {depth_cond}').index, ['Recovered', 'Secondary']] = 'True', True
            df.loc[df.query(f'Recovered == "False" and p_ratio >= 1.98 and p_ratio <= 2.02 and {t0_cond} and {depth_cond}').index, ['Recovered', 'Secondary']] = 'Aliasx2', True
            df.loc[df.query(f'Recovered == "False" and p_ratio >= 2.98 and p_ratio <= 3.02 and {t0_cond} and {depth_cond}').index, ['Recovered', 'Secondary']] = 'Aliasx3', True
            df.loc[df.query(f'Recovered == "False" and p_ratio_inv >= 1.98 and p_ratio_inv <= 2.02 and {t0_cond} and {depth_cond}').index, ['Recovered', 'Secondary']] = 'Aliasx05', True
            df.loc[df.query(f'Recovered == "False" and p_ratio_inv >= 2.98 and p_ratio_inv <= 3.02 and {t0_cond} and {depth_cond}').index, ['Recovered', 'Secondary']] = 'Aliasx03', True
            df.drop(['sec_t0', 'sec_depth'], axis=1, inplace=True)
            
        df.loc[df.query('peak_sig == 0').index, 'Recovered'] = 'False'
        df.loc[df.query('peak_depth < 0').index, 'Recovered'] = 'False'
        
        df.drop(['t0', 'per', 'depth_ratio', 'depth_ratio_inv', 't0_offset1', 't0_offset2', 'peak_t0_adj1', 'peak_t0_adj2', 't0_diff1', 't0_diff2'], axis=1, inplace=True)
        
        df.reset_index(inplace=True)
        df.set_index(['sim_batch', 'sim_num'], inplace=True)

        self.recovered = df
        
        self.recovered.sort_values(['sim_batch', 'sim_num'], inplace=True)
        self.recovered.to_csv(self.output/f'{self.sim_type}_Recovery_{self.save_suffix}.csv')
        

    def peak_mes_recovery(self, mes_threshold=0.8):
        if 'Recovered' not in self.recovered.columns:
            raise KeyError('Run dataset_recovery first!')
        
        self.recovered.loc[self.recovered['peak_mes'] < 0.8, 'Recovered'] = 'False'
        
                
    def reduced_recovery(self):
        if len(self.recovered) == 0:
            raise ValueError('Run dataset_bls first!')
        
        if 'Recovered' not in self.recovered.columns:
            raise KeyError('Run dataset_recovery first!')
        
        df = self.recovered.copy()
        
        df.query('peak_sig != 10', inplace=True)
               
        fail_df = df.query('peak_sig == 0')
        
        df.drop(fail_df.index, inplace=True)
        
        true_df = df.query('Recovered == "True"').groupby(['sim_batch', 'sim_num']).first()
        
        df.drop(true_df.index, inplace=True)
          
        alias_df1 = df[df['Recovered'].isin(['Aliasx2', 'Aliasx05'])]
        alias_df1 = alias_df1.groupby(['sim_batch', 'sim_num']).first()
        
        df.drop(alias_df1.index, inplace=True)
        
        alias_df2 = df[df['Recovered'].isin(['Aliasx3', 'Aliasx03'])]
        alias_df2 = alias_df2.groupby(['sim_batch', 'sim_num']).first()
        
        df.drop(alias_df2.index, inplace=True)
        
        pmatch_df = df.query('Recovered == "Pmatch"').groupby(['sim_batch', 'sim_num']).first()
        
        df.drop(pmatch_df.index, inplace=True)
        
        false_df = df.groupby(['sim_batch', 'sim_num']).first()
        
        self.recovered = pd.concat([fail_df, true_df, alias_df1, alias_df2, pmatch_df, false_df]).sort_index()

        self.recovered.to_csv(self.output/f'{self.sim_type}_Recovery_{self.save_suffix}_Redux.csv')
        
