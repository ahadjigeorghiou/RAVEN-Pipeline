from pathlib import Path
import numpy as np
import pandas as pd
from TrainingSet import utils
from CandidateSet import utils as cutils
from Features import utils as futils
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import chain
import multiprocessing as mp
from itertools import repeat

class SyntheticSet(object):

    def __init__(self, sim_type, sim_location='default', sim_output='default', lc_location='default', dir_style='per_target', injection_output='default', dataset_size=None, multiprocessing=None):

        if sim_location == 'default':
            self.sim_location = Path(__file__).resolve().parents[0] / 'Simulations' / sim_type / 'lcs'
        else:
            self.sim_location = Path(sim_location) / sim_type / 'lcs'
            
        if sim_output == 'default':
            self.sim_output = Path(__file__).resolve().parents[0] / 'Simulations' / sim_type      
        else:
            self.sim_output = Path(sim_output)  / sim_type
        self.sim_output.mkdir(exist_ok=True)
        
        if lc_location == 'default':
            self.lc_location = Path(__file__).resolve().parents[0] / 'Lighcurves'
        else:
            self.lc_location = Path(lc_location)
            
        if injection_output == 'default':
            self.injection_output = Path(__file__).resolve().parents[0] / 'InjectedSet' / f'{sim_type}'
        else:
            self.injection_output = Path(injection_output)
        self.injection_output.mkdir(exist_ok=True)
            
        # Load simulation parameters
        param_file = self.sim_output / f'{sim_type}_parameters.csv'
        if param_file.exists():
            self.params = pd.read_csv(param_file, index_col=['sim_batch', 'sim_num'])
        else:
            self.params = utils.read_parameters(sim_type, self.sim_location.parent)
            self.params.to_csv(param_file)
            
        init_sim_num = len(self.params) 

        # Compute the transit/eclipse depth of the simulations if not already present in the parameters
        if 'depth' not in self.params.columns:
            with ProcessPoolExecutor(max_workers=multiprocessing) as ex:
                chunksize = len(self.params) / (10*multiprocessing)
                chunksize = int(np.ceil(chunksize))
                
                results = ex.map(utils.sim_depth, repeat(sim_type),
                                 repeat(self.sim_location), 
                                 self.params.index.get_level_values('sim_batch'), 
                                 self.params.index.get_level_values('sim_num'),
                                 self.params['P'].values,
                                 self.params['tdur'].values,
                                 chunksize=chunksize)
                    
            results = list(results)
            
            depths = pd.DataFrame(results, columns=['sim_batch', 'sim_num', 'depth'])
            depths.set_index(['sim_batch','sim_num'], inplace=True)
            
            self.params = self.params.join(depths)
            
            self.params.to_csv(param_file)
            
        # Remove sims where the depth is below 200ppm
        remove = self.params.query('depth < 200 or depth != depth').index
        if len(remove) > 0:
            self.params.drop(remove, inplace=True)
            if len(remove) == 1:
                print(f'{len(remove)} simulation had depth below 200ppm and was removed!')
            else: 
                print(f'{len(remove)} simulations had depth below 200ppm and were removed!')
        
        # Remove sims where the period is less than 0.5 days
        remove = self.params.query('P < 0.5').index
        if len(remove) > 0:
            self.params.drop(remove, inplace=True)
            if len(remove) == 1:
                print(f'{len(remove)} simulation had a period shorter than 0.5 days and was removed!')
            else: 
                print(f'{len(remove)} simulations had a period shorter than 0.5 days and were removed!')
            
        # Set the parameters index to the ticid to help with injecting multiple sims to the same star    
        self.params.reset_index(inplace=True)
        self.params.set_index('ticid', inplace=True)
        
        # Load sectors information for the sim targets
        sector_file = Path(__file__).resolve().parents[1] / 'Input' / f'SPOC_sectors.csv'
        if sector_file.exists():
            sectors = pd.read_csv(sector_file, index_col='ticid')
            sectors['sector'] = sectors['sector'].apply(lambda x: [int(s) for s in x.split(',') if len(s)>0])
        else:
            raise ValueError('Sector file not found!')
          
        # Join the sectors to the parameters. 
        # This removes simulations for which no SPOC file for the target exists.
        self.params = self.params.join(sectors, how='inner')
        
        if len(self.params) != init_sim_num:
            print(f'A total of {len(self.params)} simulations remain out of an initial size of {init_sim_num} '
                'after correcting for the depth, period and existence of a SPOC target lc.')
        else:
             print(f'All {init_sim_num} have the required depth, period and a SPOC target lc.')
                
        # Downscale the sample if requested
        if dataset_size and dataset_size < len(self.params):
            self.params = self.params.sample(dataset_size)
            
        self.sim_type = sim_type
        self.dir_style = dir_style
        self.multiprocessing = multiprocessing

    def create_synthetic_lcs(self):
        ticids = self.params.index.unique()
        print(f'Injecting {len(self.params)} {self.sim_type} simulations')
        if self.multiprocessing is None or len(ticids) < 10:
            print(f'Running on a single core.')
            injection_log = []
            for ticid in ticids:
                
                logs = self.inject_sims(ticid)
                for log in logs:
                    injection_log.append(log)
        else:
            if len(ticids) < self.multiprocessing:
                workers = len(ticids)
            else:
                workers = self.multiprocessing
            
            print(f'Running on {workers} cores.')   
            with ThreadPoolExecutor(max_workers=workers) as ex:
                try:
                    
                    chunk = int(np.ceil(len(ticids) / (10*workers)))
                    results = ex.map(self.inject_sims, ticids, chunksize=chunk)
                    
                    injection_log = list(chain.from_iterable(results))
                except KeyboardInterrupt:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise ValueError('Keyboard interrupt')       

        self.injected = pd.DataFrame(injection_log, columns=['sim_batch', 'sim_num', 'ticid', 'per', 't0', 'tdur', 'depth', 'target_fraction', 'transits', 'sectors'])
        self.injected.sort_values(by=['sim_batch', 'sim_num'], inplace=True)
        self.injected.set_index(['sim_batch', 'sim_num'], inplace=True)
        self.injected.to_csv(self.sim_output / f'{self.sim_type}_InjectionLog.csv')

        
    def inject_sims(self, ticid):
        try:
            subset = self.params.loc[[ticid]]
            log = []
            
            sectors = subset.iloc[0]['sector']
            
            time = np.array([])
            
            fractions = []
        
            times = {}
            fluxes = {}
            
            exp_times = {}
            scc = ''
            sec_length = ''
            injected_sectors = ''
            
            for sec in sectors:
                # Locate and load lightcurves

                lc_file = cutils.lc_filepath(self.lc_location, self.dir_style, ticid, sec)
                
                if lc_file is None:
                    try:
                        sec_time, sec_flux, cam, ccd, fraction = utils.download_spoc_lc(ticid, sec)
                    except Exception:
                        continue
                else:
                    try:
                        sec_time, sec_flux, cam, ccd, fraction = utils.load_spoc_lc(lc_file)
                    except FileNotFoundError:
                        continue
                
                exp_time = np.nanmedian(np.diff(sec_time)) * 24 * 60 * 60
                try:
                    exp_time = int(np.round(exp_time))
                except ValueError:
                    continue
                
                times[sec] = sec_time
                fluxes[sec] = sec_flux
                exp_times[sec] = exp_time
                
                scc += f'S{sec:02}-{cam}-{ccd} '
                
                fractions.append(fraction)
                
                sec_length += f'{len(sec_time)} '
                
                injected_sectors += f'{sec},'

                time = np.concatenate([time, sec_time])

            injected_sectors = injected_sectors[:-1]
            
            mean_fraction = np.nanmean(fractions)
            
            for i in range(len(subset)):
                data = subset.iloc[i]
                        
                sim_batch = int(data['sim_batch'])
                sim_num = int(data['sim_num'])
                
                per = data['P']

                tdur = data['tdur']
                        
                if len(time) == 0:
                    t0 = None
                    depth = None
                    transits = None
                    mean_fraction = None
                    injected_sectors = ''
                    log.append((sim_batch, sim_num, ticid, per, t0, tdur, depth, mean_fraction, transits, injected_sectors))
                    continue

                t0 = utils.choose_epoch(time, per, tdur)
                
                # Get number of transits
                transits = futils.observed_transits(time, t0, per, tdur)
                
                try:
                    sim = utils.load_simulation(self.sim_location, self.sim_type, sim_batch, sim_num)
                except FileNotFoundError:
                    t0 = t0
                    depth = None
                    transits = None
                    mean_fraction = None
                    injected_sectors = ''
                    log.append((sim_batch, sim_num, ticid, per, t0, tdur, depth, mean_fraction, transits, injected_sectors))
                    continue

                depths = []
                injected_flux = np.array([])
                for sec in times.keys():
                    sim_function, interp_sim = utils.interpolate_exposure(sim.copy(), per, exp_times[sec])

                    depths.append(utils.calculate_depth(interp_sim, per, tdur))

                    phase = utils.phasefold(times[sec], per, t0 - per*0.5) - 0.5

                    sim_values = sim_function(phase)
                    
                    sec_injected_flux = np.multiply(fluxes[sec], sim_values)

                    injected_flux = np.concatenate([injected_flux, sec_injected_flux])

                depth = np.mean(depths)*1e6
                
                # Save injected_flux
                self.save_synthetic_lcs(ticid, sim_batch, sim_num, scc, sec_length, time, injected_flux, t0, per, tdur, depth)
                
                log.append((sim_batch, sim_num, ticid, per, t0, tdur, depth, mean_fraction, transits, injected_sectors))
                
            return log
        except Exception:
            return []

    def save_synthetic_lcs(self, ticid, sim_batch, sim_num, scc, sec_length, time, flux, t0, per, tdur, depth):
        hdr = fits.Header()

        hdr['TICID'] = (ticid, 'TIC ID')
        hdr['SIM'] = (f'{self.sim_type}-{sim_batch}-{sim_num}', 'Simulation ID')
        hdr['SCC'] = (scc, 'SECTOR-CAMERA-CCD')
        hdr['SECLEN'] = (sec_length, 'Sector data length')
        hdr['T0'] = (t0, 'Randomly chosen injection Epoch Time')
        hdr['P'] = (per, 'Transiting Event Period')
        hdr['TDUR'] = (tdur, 'Transit duration in days')
        hdr['DEPTH'] = (depth, 'Theoretical transit depth from simulation')
        
        primary = fits.PrimaryHDU(header=hdr)

        time_col = fits.Column(name='TIME', array=time, format='D', unit='BJD - 2457000, days', disp='D14.7')
        flux_col = fits.Column(name='FLUX', array=flux, format='E', unit='e-/s', disp='E14.7')

        table = fits.BinTableHDU.from_columns([time_col, flux_col])

        table.name = 'LIGHTCURVE'

        hdul = fits.HDUList([primary, table])
        
        output_file = self.injection_output / f'{self.sim_type}-{sim_batch}-{sim_num}.fits'

        hdul.writeto(output_file, overwrite=True)

