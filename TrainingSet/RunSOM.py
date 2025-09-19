from pathlib import Path
import pandas as pd
import numpy as np
from Features.TransitSOM import TransitSOM as tsom

def CreateSOMarrays(sim_type, lc_location, recovery_file, alias=False, save_suffix='', load_suffix=None, multiprocessing=0):
    lc_loc = Path(lc_location)
    directory = Path(__file__).resolve().parent
    output_dir =  directory / 'Simulations' / f'{sim_type}'
    
    if load_suffix is not None:
        som_array_in = np.load(output_dir / f'{sim_type}_som_array{load_suffix}.npy')
        som_ids_in = pd.read_csv(output_dir / f'{sim_type}_som_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
        som_ids_in = som_ids_in.index
    else:
        som_array_in = None
        som_ids_in = pd.MultiIndex(names=['sim_batch', 'sim_num'], levels=[[],[]], codes=[[],[]])
        
    recovery_file = output_dir / f'{recovery_file}'
    injection_log = output_dir / f'{sim_type}_InjectionLog.csv'
    rec = pd.read_csv(recovery_file).set_index(['sim_batch', 'sim_num'])
    if alias:
        rec.query('Recovered == "True" or Recovered == "Aliasx2" or Recovered == "Aliasx05"', inplace=True)
    else:
        rec.query('Recovered == "True"', inplace=True)
    rec.sort_index(inplace=True)
    
    df = pd.read_csv(injection_log).set_index(['sim_batch', 'sim_num'])
    df = df.loc[rec.index]
    
    if 'Secondary' in rec.columns:
        secondary_idx = rec.query('Secondary == True').index
        df.loc[secondary_idx, 't0'] = df.loc[secondary_idx, 'sec_t0']
        df.loc[secondary_idx, 'tdur'] = df.loc[secondary_idx, 'sec_tdur']
        df.loc[secondary_idx, 'depth'] = df.loc[secondary_idx, 'sec_depth']
            
    if alias:
        alias05 = rec.query('Recovered == "Aliasx05"')
        df.loc[alias05.index, 'per'] *= 0.5
        alias2 = rec.query('Recovered == "Aliasx2"')
        df.loc[alias2.index, 'per'] *= 2

    df = df.loc[df.index.difference(som_ids_in)]
    print(f'Processing {len(df)} {sim_type} files')
    
    files_lst = []
    for sim_batch, sim_num in df.index:
        filename = f'{sim_type}-{sim_batch}-{sim_num}.fits'
        filepath = lc_loc / f'{sim_type}' / filename
        
        files_lst.append(filepath)
        
    files_lst = np.array(files_lst)

    
    if len(files_lst) > 0:
        som_array, relative_ids_pla = tsom.PrepareLightcurves(files_lst, df['per'].values, df['t0'].values, df['tdur'].values, nbins=48, multiprocessing=multiprocessing)
        som_ids = df.index[relative_ids_pla]
        
        if som_array_in is not None:
            som_array = np.concatenate([som_array, som_array_in])
            som_ids = som_ids.union(som_ids_in, sort=False)
    elif som_array_in is not None:
        som_array = som_array_in
        som_ids = som_ids_in
    else:
        raise ValueError(f'There are no {sim_type} lightcurves!')
    
    sort_idx = som_ids.argsort()
    
    som_ids = som_ids[sort_idx]
    som_array = som_array[sort_idx]
    
    array_outfile = output_dir / f'{sim_type}_som_array{save_suffix}.npy'
    np.save(array_outfile, som_array)
    
    ids_outfile = output_dir / f'{sim_type}_som_ids{save_suffix}.csv'
    som_ids = pd.DataFrame(index=som_ids).to_csv(ids_outfile)
    
    print('SOM arrays created')
    
    
def TrainSOM(sim_type1, sim_type2, train_suffix1='training', train_suffix2='training', load_suffix='', return_som=False):
    directory = Path(__file__).resolve().parent
    output_dir1 =  directory / 'Simulations' / f'{sim_type1}'
    output_dir2 =  directory / 'Simulations' / f'{sim_type2}'
    
    som_array1 = np.load(output_dir1 / f'{sim_type1}_som_array{load_suffix}.npy')
    som_ids1 = pd.read_csv(output_dir1 / f'{sim_type1}_som_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
    som_ids1['idx'] = np.arange(len(som_ids1), dtype=int)
    
    sim1_train_ids = pd.read_csv(output_dir1 / f'{sim_type1}_{train_suffix1}_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
    
    som_array2 = np.load(output_dir2 / f'{sim_type2}_som_array{load_suffix}.npy')
    som_ids2 = pd.read_csv(output_dir2 / f'{sim_type2}_som_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
    som_ids2['idx'] = np.arange(len(som_ids2), dtype=int)
    
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
    
    idx1 = som_ids1.loc[sim1_train_ids]['idx'].values
    idx2 = som_ids2.loc[sim2_train_ids]['idx'].values
    
    som_array1 = som_array1[idx1]
    som_array2 = som_array2[idx2]
    
    som_array = np.concatenate([som_array1, som_array2])
    
    outfile = directory / 'Output' / f'{sim_type1}-{sim_type2}_SOM{load_suffix}.txt'
    print(f'Training a SOM for {len(som_array1)} {sim_type1} and {len(som_array2)} {sim_type2}')
    if return_som:
        som = tsom.CreateSOM(som_array, outfile=outfile, return_som=True)
    
        return som
    else:
        tsom.CreateSOM(som_array, outfile=outfile, return_som=False)
     
    
def load_som_array(directory, sim_type, load_suffix):
    directory = Path(directory)
    som_array = np.load(directory / f'{sim_type}_som_array{load_suffix}.npy')
    som_ids = pd.read_csv(directory / f'{sim_type}_som_ids{load_suffix}.csv').set_index(['sim_batch', 'sim_num'])
    som_ids['idx'] = np.arange(len(som_ids), dtype=int)
    
    return som_array, som_ids


def load_training_som_array(sim_type1, sim_type2, sim1_train_ids, sim2_train_ids, load_suffix='', directory=None):
    if directory is None:
        directory = Path(__file__).resolve().parent / 'Simulations'
    else:
        directory = Path(directory)
        
    sim_dir1 =  directory / f'{sim_type1}'
    sim_dir2 =  directory / f'{sim_type2}'
    
    som_array1, som_ids1 = load_som_array(sim_dir1, sim_type1, load_suffix)

    som_array2, som_ids2 = load_som_array(sim_dir2, sim_type2, load_suffix)
    
    # Use only the som arrays corresponding to the training set ids
    idx1 = som_ids1.loc[sim1_train_ids]['idx'].values
    idx2 = som_ids2.loc[sim2_train_ids]['idx'].values
    
    som_array1 = som_array1[idx1]
    som_array2 = som_array2[idx2]
    
    # Construct the full training som array
    som_array = np.concatenate([som_array1, som_array2])
    
    return som_array
        
        
    