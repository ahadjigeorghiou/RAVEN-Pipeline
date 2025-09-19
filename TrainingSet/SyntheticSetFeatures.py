from pathlib import Path
import numpy as np
import pandas as pd
from TrainingSet import utils
from Features import featurecalc as fc
from Features import utils as futils
from concurrent.futures import ProcessPoolExecutor, as_completed
from Features.TransitSOM import TransitSOM
import pickle
from TrainingSet.RunSOM import load_training_som_array


class SyntheticSet(object):

    def __init__(self, sim_type, input_loc='default', output_loc='default', lc_loc='default', recovery_name=None, size=None, alias=False, save_sample=False, multiprocessing=None):

        self.sim_type = sim_type
        
        if input_loc == 'default':
            self.input_loc = Path(__file__).resolve().parents[1] / 'TrainingSet' / 'Simulations' / sim_type
        else:
            self.input_loc = Path(input_loc) / sim_type
                    
        if output_loc == 'default':
            self.output = self.input_loc
        else:
            self.output = Path(output_loc)
            
        if lc_loc == 'default':
            self.lc_loc = Path(__file__).resolve().parents[0] / 'InjectedSet' / f'{sim_type}'
        else:
            self.lc_loc = Path(lc_loc) / sim_type
        
        param_file = self.input_loc / f'{sim_type}_parameters.csv'
        try:
            self.params = pd.read_csv(param_file).set_index(['sim_batch', 'sim_num'])
        except FileNotFoundError:
            raise ValueError(f'Simulation parameters file for the synthetic lc set not found at location: {self.input_loc}')
        
        injection_file = self.input_loc / f'{sim_type}_InjectionLog.csv'
        try:
            self.injection_log = pd.read_csv(injection_file).set_index(['sim_batch', 'sim_num'])
        except FileNotFoundError:
            raise ValueError(f'Simulation injection file for the synthetic lc set not found at location: {self.input_loc}')
        
        try:
            if recovery_name is None:
                self.recovered = pd.read_csv(self.input_loc / f'{sim_type}_Recovery_Redux.csv').set_index(['sim_batch', 'sim_num'])
            else:
                self.recovered = pd.read_csv(self.input_loc / f'{recovery_name}').set_index(['sim_batch', 'sim_num'])
        except FileNotFoundError:
            raise ValueError(f'Recovery file for the synthetic lc set not found at location: {self.input_loc}')
        
        if len(self.recovered) > len(self.injection_log):
            raise ValueError('Size of BLS Recovery output does not match the number of Injected LCs!') 
        
        if alias:
            self.recovered.query('Recovered == "True" or Recovered == "Aliasx2" or Recovered == "Aliasx05"', inplace=True)
        else:
            self.recovered.query('Recovered == "True"', inplace=True)
        
        if len(self.recovered) == 0:
            raise ValueError('There are no recovered events!')
           
        if size is not None and len(self.recovered) > size:
            self.recovered = self.recovered.sample(size)
            self.recovered.sort_index(inplace=True)
            if save_sample:
                self.recovered.to_csv(self.input_loc / f'{sim_type}_Recovery_Sample_{size}.csv') 
                
        self.injection_log = self.injection_log.loc[self.recovered.index]
        self.params = self.params.loc[self.recovered.index]
        
        # Computed the combined Tmag, Gmag and BP-RP for the simulations and add them to the parameters.
        if sim_type not in ['NEB', 'NTRIPLE', 'NPLA', 'NuPLA']:
            self.params = utils.add_stellar_params(sim_type, self.params)
        else:
            self.params.rename(columns={'target_Tmag':'Tmag',
                                        'target_Gmag':'Gmag',
                                        'target_BP-RP':'BP-RP'}, inplace=True)
        
        if alias:
            alias05 = self.recovered.query('Recovered == "Aliasx05"')
            self.injection_log.loc[alias05.index, 'per'] *= 0.5
            alias2 = self.recovered.query('Recovered == "Aliasx2"')
            self.injection_log.loc[alias2.index, 'per'] *= 2

        if 'Secondary' in self.recovered.columns:
            secondary_idx = self.recovered.query('Secondary == True').index
            self.injection_log.loc[secondary_idx, 't0'] = self.injection_log.loc[secondary_idx, 'sec_t0']
            self.injection_log.loc[secondary_idx, 'tdur'] = self.injection_log.loc[secondary_idx, 'sec_tdur']
            self.injection_log.loc[secondary_idx, 'depth'] = self.injection_log.loc[secondary_idx, 'sec_depth']
                        
        self.recovered['sec_num'] = self.recovered['sec_num'].astype(int)
        self.recovered['peak_sig'] = self.recovered['peak_sig'].astype(int)

        self.multiprocessing = multiprocessing
        
        self.features = pd.DataFrame()
        
        
    def generate_features(self, rerun=False, save_suffix='', load_suffix=None):
        # Load in data from file, if provided at class initialisation or when the function was called.
        if load_suffix is not None and not rerun:
            infile = self.output / f'{self.sim_type}_features{load_suffix}.csv'
            print('Loading from ' + str(infile))
            try:
                self.features = pd.read_csv(infile).set_index(['sim_batch', 'sim_num'])
            except Exception as e:
                # Handle failure to load existing data
                print(e)
                print('Error loading existing features, recreating...')

        
        to_run = self.recovered.index.difference(self.features.index)
        num = len(to_run)
        print(f'Producing features for {num} out of {len(self.recovered)} synthetic {self.sim_type} events.')
        
        if num > 0:
            if self.multiprocessing > 1 and num > 5:
                if num < self.multiprocessing:
                    # If there are less targets than the specified number of cores, set the number of workers accordingly
                    workers = num
                else:
                    workers = self.multiprocessing
                
                print(f'Running multiprocessing on {workers} processors.')
                with ProcessPoolExecutor(max_workers=self.multiprocessing) as ex:
                    # Split the data into chunks based on the number of workers, to aid multiprocessing performance
                    factor = 20
                    while num < 5*factor*workers:
                        factor -= 1
                        if factor == 1:
                            break
                    
                    groups = np.array_split(to_run, factor*workers)
                            
                    df_lst = []
                    
                    try:
                        futures = {ex.submit(self.multi_features, group): group for group in groups}
                        
                        for future in as_completed(futures):
                            # Handle the results as they are completed. 
                            try:
                                group_features, fails = future.result()
                                df_lst.append(group_features)
                                if len(fails) > 0:
                                    # For individual exceptions on a target, explicitly caught and handled in the code
                                    for fail in fails:
                                        print(f'Exception {fail[2]} occur with sim id: {fail[0]}-{fail[1]}')
                            except Exception as e:
                                # For exceptions that were not caught and handled in the code, which lead to failure for the whole group of ticids
                                group = futures[future]
                                print(f'Exception {e} occur with ticid group: {group}')
                    except KeyboardInterrupt:
                        # Attempt to save and shutdown multi-processed work, if interrupted
                        df_lst.append(self.features)
                        self.features = pd.concat(df_lst)
                        try:
                            self.features.sort_values(['sim_batch', 'sim_num'], inplace=True)
                            self.features.to_csv(self.output / f'{self.sim_type}_features{save_suffix}.csv')
                        except KeyError:
                            print('Features were not computed!')
                        
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')    
            else: 
                # Run on a single core
                groups = np.array_split(to_run, int(np.ceil(num/10)))
                df_lst = []
                
                for group in groups:
                    features, fails = self.multi_features(group)
                    
                    df_lst.append(features)
                    
                    if len(fails) > 0:
                        # For individual exceptions on a target, explicitly caught and handled in the code
                        for fail in fails:
                            print(f'Exception {fail[2]} occur with sim id: {fail[0]}-{fail[1]}')
                    
            # Save features
            df_lst.append(self.features)
            self.features = pd.concat(df_lst)
            try:
                self.features.sort_values(['sim_batch', 'sim_num'], inplace=True)
                self.features.to_csv(self.output / f'{self.sim_type}_features{save_suffix}.csv')
            except KeyError:
                print('Features were not computed!')
       
            
    def multi_features(self, sim_indices):
        feat_lst = []
        failed = []
        for sim_batch, sim_num in sim_indices:
            try:
                features_df = self.single_sim_features(sim_batch, sim_num)
                feat_lst.append(features_df)
            except Exception as e:
                failed.append((sim_batch, sim_num, e))
        
        try:    
            features_df = pd.concat(feat_lst)
        except ValueError:
            features_df = pd.DataFrame()
        
        return features_df, failed
            
            
    def single_sim_features(self, sim_batch, sim_num):
        sim_params = self.params.loc[(sim_batch, sim_num)]
        
        ticid = int(sim_params['ticid'])
        rstar = sim_params['target_R']
        teff = sim_params['target_teff']
        tmag = sim_params['Tmag']
        gmag = sim_params['Gmag']
        dist = sim_params['target_dist']
        bp_rp = sim_params['BP-RP']
        
        per, t0, tdur, depth = self.injection_log.loc[sim_batch, sim_num][['per', 't0', 'tdur', 'depth']]
        
        if tdur/per > 0.35:
            tdur = 0.33*per
            
        file = self.lc_loc / f'{self.sim_type}-{sim_batch}-{sim_num}.fits'
        lc, sec_lcs = utils.load_injected_lc(file, winsize=2.0, transitcut=True, t0=t0, per=per, tdur=tdur)
        
        lc['lcs'] = sec_lcs
                
        features = {}
        
        features['sim_batch'] = sim_batch
        features['sim_num'] = sim_num
        features['ticid'] = ticid
        features['Tmag'] = tmag
        features['Gmag'] = gmag
        features['BP-RP'] = bp_rp
        features['dist'] = dist
        features['rstar'] = rstar
        features['teff'] = teff
        features['per'] = per
        features['t0'] = t0
        features['tdur'] = tdur
        features['tdur_per'] = tdur/per
        features['depth'] = depth
        
        features['disp'] = self.sim_type
        
        obs_transits = futils.observed_transits(lc['time'], t0, per, tdur)

        features['transits'] = obs_transits
        
        features = fc.generate_LCfeatures(lc, features, per, t0, tdur, depth, rstar, teff)
        
        if self.sim_type in ['NEB', 'NTRIPLE', 'NPLA']:
            features['target_fraction'] = sim_params['target_fraction']
            features['nearby_fraction'] = sim_params['nearby_fraction']
        
        features_df = pd.DataFrame(features, index=[0]).set_index(['sim_batch', 'sim_num'])
        
        return features_df
        
        
