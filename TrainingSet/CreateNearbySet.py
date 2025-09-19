from pathlib import Path
import numpy as np
import pandas as pd
from TrainingSet import utils
from Features import utils as futils
from CandidateSet import utils as cutils
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool
from itertools import chain, repeat
from scipy.stats import gaussian_kde
import warnings
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter('ignore', category=VerifyWarning)

def compute_kde_reflected(fractions, bw):
    reflected = np.concatenate([fractions, 1-fractions+1])
    
    kde = gaussian_kde(reflected, bw_method=bw)
    
    return kde


def sample_kde(kde, num, lower_limit, reflected=True):
    if reflected == True:
        samples = kde.resample(int(num*2.5))
    else:
        samples = kde.resample(num*1.5)
    
    samples = samples[(samples >= lower_limit) & (samples <= 1.0)]
    
    samples = samples[:num]
    
    return samples
    
        
class SyntheticSet(object):

    def __init__(self, sim_type, sim_location='default', sim_output='default', lc_location='default', dir_style='per_target', injection_output='default', sector_file='SPOC_sectors.csv', sim_input='sim_input.csv', dataset_size=None, multiprocessing=None):
        
        self.sim_type = sim_type
        self.dataset_size = dataset_size    
        self.dir_style = dir_style 
        self.multiprocessing = multiprocessing
         
        if sim_location == 'default':
            self.sim_location = Path(__file__).resolve().parents[0] / 'Simulations' / sim_type
        else:
            self.sim_location = Path(sim_location) / sim_type / 'lcs' 
            
        if sim_output == 'default':
            self.primary_sim_loc = Path(__file__).resolve().parents[0] / 'Simulations' / sim_type
            self.sim_output = Path(__file__).resolve().parents[0] / 'Simulations' / f'N{sim_type}'      
        else:
            self.sim_output = Path(sim_output)  / f'N{sim_type}'
            self.primary_sim_loc = Path(sim_output) / sim_type
        
        self.primary_sim_loc.mkdir(exist_ok=True)
        self.sim_output.mkdir(exist_ok=True)
        
        if lc_location == 'default':
            self.lc_location = Path(__file__).resolve().parents[0] / 'Lighcurves'
        else:
            self.lc_location = Path(lc_location)
            
        if injection_output == 'default':
            self.injection_output = Path(__file__).resolve().parents[0] / 'InjectedSet' / f'N{sim_type}'
        else:
            self.injection_output = Path(injection_output) / f'N{sim_type}'
        self.injection_output.mkdir(exist_ok=True)
            
        # Load primary simulation parameters
        param_file = self.primary_sim_loc / f'{sim_type}_parameters.csv'
        if param_file.exists():
            self.params = pd.read_csv(param_file).set_index(['sim_batch', 'sim_num'])
        else:
            self.params = utils.read_parameters(sim_type, self.sim_location)
            self.params.to_csv(param_file)
            
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

        # Load sectors information
        sector_file = sector_file = Path(__file__).resolve().parents[1] / 'Input' / sector_file

        self.sectors = pd.read_csv(sector_file, index_col='ticid')
        
        self.target_fractions = pd.read_csv(Path(__file__).resolve().parents[0] / 'TargetFractions_All55.csv', index_col='ticid')
        self.target_fractions.query('mean != 0', inplace=True)
        self.nearby_fractions = pd.read_csv(Path(__file__).resolve().parents[0] / 'NearbyFractions.csv', index_col='ticid')
        self.nearby_fractions.query('new_nearby_ratio == new_nearby_ratio', inplace=True)
        
        input_file = input_file = Path(__file__).resolve().parents[1] / 'Input' / sim_input
        self.sim_input = pd.read_csv(input_file, index_col='ticid')
        
        self.sim_input = self.sim_input.join(self.target_fractions['mean'], how='inner')
        self.sim_input = self.sim_input[['Teff','mass','logg','Rad','Tmag','Gmag','distance', 'BP-RP', 'mean']]
        
        self.sim_input.rename(columns={'Teff':'target_teff',
                                        'mass':'target_mass',
                                        'logg':'target_logg',
                                        'Rad':'target_R',
                                        'distance':'target_dist',
                                        'Tmag':'target_Tmag',
                                        'Gmag':'target_Gmag',
                                        'BP-RP':'target_BP-RP',
                                        'mean':'target_fraction'}, inplace=True)
        
        self.sectors = self.sectors.loc[self.sectors.index.intersection(self.sim_input.index)]
        # self.target_fractions = self.target_fractions.loc[self.target_fractions.index.intersection(self.sim_input.index)]
        self.nearby_fractions = self.nearby_fractions.loc[self.nearby_fractions.index.intersection(self.sim_input.index)]
                        
        # self.nearby_params = None
        
        
    def create_nearby_params_new(self, load_existing=True):
        if load_existing:
            try:
                self.nearby_params = pd.read_csv(self.sim_output / f'N{self.sim_type}_parameters.csv', index_col=['sim_batch','sim_num'])
                self.rejection_params = pd.read_csv(self.sim_output / f'N{self.sim_type}_rejections.csv')
            except FileNotFoundError:
                self.nearby_params = None
                self.rejection_params = None
        else:
            self.nearby_params = None
            self.rejection_params = None
            
        if self.nearby_params is None:
            self.params['sim_name'] = [f'{self.sim_type}-{b}-{n}' for b,n in self.params.index]
            
            sim_input = self.sim_input.copy()
            sim_input.reset_index(inplace=True)
            sim_input = sim_input.sample(len(sim_input))
            
            sim_input['nearby_fraction_ratio'] = self.nearby_fractions['new_nearby_ratio'].sample(len(sim_input), replace=True).values
            sim_input['nearby_fraction'] = (1-sim_input['target_fraction'])*sim_input['nearby_fraction_ratio']
            
            host_sample = self.params.sample(len(sim_input), replace=True)
            
            sim_input['sim_name'] = host_sample['sim_name'].values
            sim_input['P'] = host_sample['P'].values
            sim_input['tdur'] = host_sample['tdur'].values
            sim_input['ecc'] = host_sample['ecc'].values
            sim_input['omega'] = host_sample['omega'].values
            sim_input['incl'] = host_sample['incl'].values
            sim_input['b'] = host_sample['b'].values
            sim_input['depth'] = host_sample['depth'].values
            
            sim_input['diluted_depth'] = sim_input['depth'] * sim_input['nearby_fraction']/sim_input['target_fraction']
            sim_input['Valid'] = 'True'
            sim_input.loc[sim_input['diluted_depth'] > 1e6, 'Valid'] = 'Above'
            sim_input.loc[sim_input['diluted_depth'] < 300, 'Valid'] = 'Below'
            
            self.nearby_params = sim_input.query('Valid == "True"').copy()
            self.rejection_params = sim_input.query('Valid == "Above" or Valid == "Below"').copy()
            
            for i, idx in enumerate(np.split(self.nearby_params.index, np.arange(10000,len(self.nearby_params),10000))):
                self.nearby_params.loc[idx, 'sim_batch'] = i
                self.nearby_params.loc[idx, 'sim_num'] = np.arange(len(idx), dtype=int)
            
            self.nearby_params['sim_batch'] = self.nearby_params['sim_batch'].astype(int)
            self.nearby_params['sim_num'] = self.nearby_params['sim_num'].astype(int)
            
            self.nearby_params.set_index(['sim_batch', 'sim_num'], inplace=True)
            
            self.nearby_params.to_csv(self.sim_output / f'N{self.sim_type}_parameters.csv')
            self.rejection_params.to_csv(self.sim_output / f'N{self.sim_type}_rejections.csv', index=False)
        
        
    def create_nearby_params(self, bw=0.08, sample_size=1000000, reuse_output=False):
        if reuse_output:
            input_params = pd.read_csv(self.sim_output / f'N{self.sim_type}_parameters.csv')
            input_params.drop(['sim_batch', 'sim_num'], axis=1, inplace=True)
            with open(self.sim_output / f'N{self.sim_type}_fails.txt', 'r') as f:
                text = f.readlines()
            below = int(text[0].split(' ')[3])
            above = int(text[1].split(' ')[3])
            count = len(input_params) + below + above
        else:
            below = 0
            above = 0
            count = 0
            input_params = pd.DataFrame()
        
        if len(input_params) > 0:
            dataset_size = self.dataset_size - len(input_params)
        else:
            dataset_size = self.dataset_size
            
        if dataset_size <= 0:
            sample_size = 0
            print(f'Already run parameters fulfil the required dataset size. No new parameters will be created.')
            
            
        if len(self.nearby_fractions) < 50000:   
            print('Sampling from KDE.') 
            kde = compute_kde_reflected(self.nearby_fractions['nearby'].values, bw)
            
            nearby_samples = sample_kde(kde, sample_size, 0.02, reflected=True)
        else:
            print('Sampling directly from the nearby population.') 
            nearby_samples = self.nearby_fractions['new_nearby_ratio'].sample(sample_size, replace=True).values
        
        results = []
        
        print(f'Computing parameters for {dataset_size} N{self.sim_type} out of {sample_size} sample fractions')
        if self.multiprocessing > 1 and sample_size > 10:
            if self.multiprocessing > sample_size:
                workers = sample_size
            else:
                workers = self.multiprocessing
                
            print(f'Running on {workers} cores')
            with Pool(workers) as p:
                chunksize = int(np.ceil(sample_size/(workers*15)))
                try:
                    for result in p.imap(self.estimate_nearby_depth, nearby_samples, chunksize=chunksize):
                        if result == 'Below':
                            below += 1
                        elif result == 'Above':
                            above += 1
                        else:
                            results.append(result)
                        
                        count += 1
                        if len(results) >= dataset_size:
                            break
                        
                except KeyboardInterrupt:
                    print('Interrupted. Attempting to save...')
        else:
            for nb_fraction in nearby_samples:
                try:
                    result = self.estimate_nearby_depth(nb_fraction)
                    if result == 'Below':
                        below += 1
                    elif result == 'Above':
                        above += 1
                    else:
                        results.append(result)
                    
                    if len(results) >= dataset_size:
                        break
                except KeyboardInterrupt:
                    print('Interrupted. Attempting to save...')
        
        print(f'Parameter creation completed.')
        columns = ['sim_name', 'target_fraction', 'nearby_fraction', 'P', 'tdur', 'ecc', 'omega', 'incl', 'diluted_depth', 'ticid']            
        self.nearby_params = pd.concat([input_params, pd.DataFrame(results, columns=columns)], ignore_index=True)
        
        # To maintain compatibility with the rest of the simulated scenarios, 
        # set a new sim batch and number for each nearby sim
        try:
            for i, idx in enumerate(np.array_split(self.nearby_params.index, 10)):
                self.nearby_params.loc[idx, 'sim_batch'] = i
                self.nearby_params.loc[idx, 'sim_num'] = np.arange(len(idx), dtype=int)
            self.nearby_params.set_index(['sim_batch', 'sim_num'], inplace=True)
        except ValueError:
            print('Setting sim_batch and sim_num failed! Creating params without.')
        
        # Save the new parameters file
        self.nearby_params.to_csv(self.sim_output / f'N{self.sim_type}_parameters.csv')
        # Save the number of diluted sims which resulted in a depth outside the limits
        with open(self.sim_output / f'N{self.sim_type}_fails.txt', 'w') as f:
            f.write(f'Below 300ppm - {below} \n')
            f.write(f'Above 1e6ppm - {above}')
        
                               
    def estimate_nearby_depth(self, nearby_fraction):
        target_fraction = self.target_fractions.sample(1)
        ft = target_fraction['mean'].values[0]
        
        fnb = (1-ft)*nearby_fraction
        
        sim_params = self.params.sample(1)

        sim_batch, sim_num = sim_params.index.values[0]
        sim = utils.load_simulation(self.sim_location, self.sim_type, sim_batch, sim_num)
       
        per, tdur, ecc, omega, incl = sim_params[['P', 'tdur', 'ecc', 'omega', 'incl']].values[0]
        depth = utils.calculate_depth(sim, per, tdur)
        
        diluted_depth = depth * fnb/ft
        
        diluted_depth *= 1e6
        
        if diluted_depth < 300:
            return 'Below'
        elif diluted_depth > 1e6:
            return 'Above'
        else:
            sim_name = f'{self.sim_type}-{sim_batch}-{sim_num}'
            ticid = target_fraction.index.values[0]
            
            return (sim_name, ft, fnb, per, tdur, ecc, omega, incl, diluted_depth, ticid)
    
    def create_synthetic_lcs(self, dataset_size=100000, reuse_output=False):
        if self.nearby_params is None:
            # Try to read the nearby parameters file
            try:
                self.nearby_params = pd.read_csv(self.sim_output / f'N{self.sim_type}_parameters.csv').set_index(['sim_batch', 'sim_num'])
            except FileNotFoundError:
                raise ValueError('Run create_nearby_params first!')
        
        if reuse_output:
            infile = self.sim_output / f'N{self.sim_type}_InjectionLog.csv'
            try:
                input_df = pd.read_csv(infile).set_index(['sim_batch', 'sim_num'])
            except FileNotFoundError:
                print(f'Output file: {infile} not found!')
                input_df = pd.DataFrame()
        else:
            input_df = pd.DataFrame()
            
        self.nearby_to_inject = self.nearby_params.loc[self.nearby_params.index.difference(input_df.index)].copy()
        
        if len(input_df) > 0:
            dataset_size -= len(input_df)
        
        if len(self.nearby_to_inject) >  dataset_size:  
            self.nearby_to_inject = self.nearby_to_inject.sample(dataset_size)
        
        self.nearby_to_inject = self.nearby_to_inject.reset_index().set_index('ticid')
        
        ticids = self.nearby_to_inject.index.unique()

        print(f'Injecting {len(self.nearby_to_inject)} N{self.sim_type} simulations')    
        if self.multiprocessing is None or len(ticids) < 10:
            print('Running on a single core.')
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
                    chunk = int(np.ceil(len(ticids) / (20*workers)))
                    results = ex.map(self.inject_sims, ticids, chunksize=chunk)
                    
                    injection_log = list(chain.from_iterable(results))
                except KeyboardInterrupt:
                    ex.shutdown(wait=False, cancel_futures=True) 
                    raise ValueError('Keyboard interrupt')       

        self.injected = pd.DataFrame(injection_log, columns=['sim_batch', 'sim_num', 'ticid', 'per', 't0', 'tdur', 'depth', 'transits', 'sectors'])
        self.injected.set_index(['sim_batch', 'sim_num'], inplace=True)
        if len(input_df) > 0:
            self.injected = pd.concat([input_df, self.injected])
        
        self.injected.sort_index(inplace=True)
        self.injected.to_csv(self.sim_output / f'N{self.sim_type}_InjectionLog.csv')

        
    def inject_sims(self, ticid):
        subset = self.nearby_to_inject.loc[[ticid]]
        log = []
        
        tic_sectors = self.sectors.loc[ticid, 'sector']
        tic_sectors = tic_sectors.split(',')[:-1]
        tic_sectors = [int(s) for s in tic_sectors]
            
        time = np.array([])
     
        times = {}
        fluxes = {}
        
        exp_times = {}
        sec_fractions = {}
        scc = ''
        sec_length = ''
        injected_sectors = ''
        
        for sec in tic_sectors:
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
                except Exception:
                    try:
                        sec_time, sec_flux, cam, ccd, fraction = utils.download_spoc_lc(ticid, sec)
                    except Exception:
                        continue
                        
            exp_time = np.nanmedian(np.diff(sec_time)) * 24 * 60 * 60
            try:
                exp_time = int(np.round(exp_time))
            except ValueError:
                continue
            
            times[sec] = sec_time
            fluxes[sec] = sec_flux
            exp_times[sec] = exp_time
            sec_fractions[sec] = fraction
            
            scc += f'S{sec:02}-{cam}-{ccd} '
            
            sec_length += f'{len(sec_time)} '
            
            injected_sectors += f'{sec},'

            time = np.concatenate([time, sec_time])

        injected_sectors = injected_sectors[:-1]
        
        for i in range(len(subset)):
            data = subset.iloc[i]
                    
            sim_batch = int(data['sim_batch'])
            sim_num = int(data['sim_num'])
            
            tf = data['target_fraction']
            nbf = data['nearby_fraction']
            nbf_percentage = nbf/(1-tf)
            per = data['P']
            tdur = data['tdur']
            depth = data['diluted_depth']
                     
            if len(time) == 0:
                t0 = None
                depth = None
                transits = None
                injected_sectors = ''
                log.append((sim_batch, sim_num, ticid, per, t0, tdur, depth, transits, injected_sectors))
                continue

            t0 = utils.choose_epoch(time, per, tdur)
            
            # Get number of transits
            transits = futils.observed_transits(time, t0, per, tdur)
            
            primary_sim = data['sim_name']
            prim_sim_batch = int(primary_sim.split('-')[1])
            prim_sim_num = int(primary_sim.split('-')[2])
            
            sim = utils.load_simulation(self.sim_location, self.sim_type, prim_sim_batch, prim_sim_num)

            injected_flux = np.array([])
            for sec in times.keys():
                sim_copy = sim.copy()
                
                ft = sec_fractions[sec]
                fnb = (1-ft)*nbf_percentage
                
                sim_copy = (sim_copy-1)*(fnb/ft) + 1
                
                sim_function, interp_sim = utils.interpolate_exposure(sim_copy, per, exp_times[sec])

                phase = utils.phasefold(times[sec], per, t0 - per*0.5) - 0.5

                sim_values = sim_function(phase)
                
                sec_injected_flux = np.multiply(fluxes[sec], sim_values)

                injected_flux = np.concatenate([injected_flux, sec_injected_flux])

            
            # Save injected_flux
            self.save_synthetic_lcs(ticid, sim_batch, sim_num, scc, sec_length, time, injected_flux, t0, per, tdur, depth)
            
            log.append((sim_batch, sim_num, ticid, per, t0, tdur, depth, transits, injected_sectors))
               
        return log

    def save_synthetic_lcs(self, ticid, sim_batch, sim_num, scc, sec_length, time, flux, t0, per, tdur, depth):
        hdr = fits.Header()

        hdr['TICID'] = (ticid, 'TIC ID')
        hdr['SIM'] = (f'N{self.sim_type}-{sim_batch}-{sim_num}', 'Simulation ID')
        hdr['SCC'] = (scc, 'SECTOR-CAMERA-CCD')
        hdr['SECLEN'] = (sec_length, 'Sector data length')
        hdr['T0'] = (t0, 'Randomly chosen injection Epoch Time')
        hdr['P'] = (per, 'Transiting Event Period')
        hdr['TDUR'] = (tdur, 'Transit duration in days')
        hdr['DEPTH'] = (depth, 'Mean transit depth from simulation')
        
        primary = fits.PrimaryHDU(header=hdr)

        time_col = fits.Column(name='TIME', array=time, format='D', unit='BJD - 2457000, days', disp='D14.7')
        flux_col = fits.Column(name='FLUX', array=flux, format='E', unit='e-/s', disp='E14.7')

        table = fits.BinTableHDU.from_columns([time_col, flux_col])

        table.name = 'LIGHTCURVE'

        hdul = fits.HDUList([primary, table])
        
        output_file = self.injection_output / f'N{self.sim_type}-{sim_batch}-{sim_num}.fits'

        hdul.writeto(output_file, overwrite=True)

