from pathlib import Path
import pandas as pd
from TrainingSet import utils

def main(sim_directory, suffix=''):
    directory = Path(sim_directory)
    script_loc = Path(__file__).resolve().parents[0]
    results = []
    for sim in ['PLA','PIB','BTP','EB','BEB','TRIPLE','uPLA','uPIB','uBTP']:    
        sim_directory =  directory / f'{sim}'
        try:
            reject_df = pd.read_csv(script_loc / f'Simulations/{sim}/{sim}_rejections.csv', index_col=['sim_batch','sim_num'])
        except FileNotFoundError:
            reject_df = utils.read_sim_rejections(sim, sim_directory)
            reject_df.to_csv(script_loc/f'Simulations/{sim}/{sim}_rejections.csv')

        try:
            params = pd.read_csv(script_loc / f'Simulations/{sim}/{sim}_parameters.csv', index_col=['sim_batch','sim_num'])
        except FileNotFoundError:
            params = utils.read_parameters(sim, sim_directory)
            params.to_csv(script_loc/f'Simulations/{sim}/{sim}_parameters.csv')
        
        reject_df.query('P >= 0.5', inplace=True)
        params.query('P >= 0.5', inplace=True)
        
        remove = params.query('depth < 200 or depth != depth').index
        params.drop(remove, inplace=True)
        
        incl = len(reject_df.query('fail == "inclination"'))
        mag = len(reject_df.query('fail == "brightness"'))
        depth = len(reject_df.query('fail == "depth"'))
        iso = len(reject_df.query('fail == "isochrone"'))
        ebop = len(reject_df.query('fail == "ebop"'))
        
        depth += len(remove)
        
        simmed = len(params)
        results.append((sim, incl, mag, depth, iso, ebop, simmed))
        
    for sim in ['NPLA','NuPLA','NEB','NTRIPLE']:
        incl = 0
        mag = 0
        iso = 0
        ebop = 0
        
        try:
            reject_df = pd.read_csv(script_loc / f'Simulations/{sim}/{sim}_rejections.csv')
        except FileNotFoundError:
            print('Create nearby params first!')
            return
        
        depth = len(reject_df)
        simmed = len(pd.read_csv(f'TrainingSet/Simulations/{sim}/{sim}_parameters.csv'))
        results.append((sim, incl, mag, depth, iso, ebop, simmed))
    
    df = pd.DataFrame(results, columns=['sim', 'incl', 'mag', 'depth', 'isochrones', 'ebop', 'simmed'])
    
    df['total'] = df['incl'] + df['mag'] + df['depth'] + df['simmed']
    
    df['sim_ratio'] = df['simmed'] / df['total']
    
    injected = []
    recovered = []
    for sim in df['sim']:
        inj = len(pd.read_csv(F'TrainingSet/Simulations/{sim}/{sim}_InjectionLog.csv'))
        rec = pd.read_csv(F'TrainingSet/Simulations/{sim}/{sim}_Recovery{suffix}_Redux.csv')
        rec = rec.query('Recovered == "True" or Recovered == "Aliasx05" or Recovered == "Aliasx2"')
        rec = len(rec)
        injected.append(inj)
        recovered.append(rec)
        
    df['injected'] = injected
    df['recovered'] = recovered
    df['recovery'] = df['recovered'] / df['injected']
    
    df['detection'] = df['sim_ratio'] * df['recovery']
    
    df.set_index('sim', inplace=True)
    
    df['adj_detection'] = df['detection']
    
    for sim in ['PLA', 'EB', 'TRIPLE', 'uPLA']:
        df.loc[f'N{sim}', 'adj_detection'] = df.loc[f'N{sim}', 'detection'] * df.loc[sim, 'sim_ratio']
        
    df.to_csv(f'TrainingSet/Output/detection_factors{suffix}.csv')
    
    
    
    
