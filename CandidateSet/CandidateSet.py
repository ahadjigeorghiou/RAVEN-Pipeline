import numpy as np
import pandas as pd
from pathlib import Path
import os
import warnings

import astropy.units as u
from copy import copy
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from CandidateSet import utils as cutils
from CandidateSet import ML_Classification
from Priors import priorutils 
from Features import utils as futils
from Features import featurecalc as fc
from Features.TransitSOM import TransitSOM as TSOM
from TrainingSet import utils as tutils

try:
    from TrainingSet import gpu_bls_new as gbls
except Exception:
    no_bls = True


class CandidateSet(object):

    def __init__(self, infile, infile_type='default', sector_file='SPOC_sectors.csv', lc_dir='default', dir_style='per_target', per_lim=[0, None], depth_lim=None, multiprocessing=1, save_output=True, save_suffix=None, load_suffix=None, plot_centroid=False):
        """
        Load in a set of candidates with their transit parameters [period, epoch, depth]. Sets up the environment for running the positional probabilitiy generation.         
        
        Parameters
        infile - path to input file or dataframe
        infile_type - options: default/archive/exofop/recovery/dataframe, allows for loading in data from specific databases or the default loading format.
        sector_file - csv file with the ticids and sectors for SPOC dataproducts
        lc_dir - either path to lighcurve directory or set as default.
        dir_style - 'per_target/'spoc'/'single. Structure of the lightcurve directory.
                    'per_target': One sub-folder per ticid which includes all available lightcurves for the target.
                    'spoc': Lightcurve directory organised as per the released spoc sector FFI data products.
                    'single': A signle directory containing all lightcurves for all targets.
        per_lim - set maximum period limit for candidates to be processed. Candidates with period longer than maximum will be skipped.
        depth_lim - set minimum depth limit for candidates. Candidates with depth less than the minimum will be skipped.
        multiprocessing - set maximum number of workers (set 1 for no multiprocessing)
        save_output - True/False. Affects all data generation except for the probability generation, which always saves the output.
        save_suffix - Suffix for the filenames of all saved data.
        load_suffix - Suffix for loading previously saved data.
        plot_centroid - True/False. Create plots when fitting the trapezium transit model to the centroid data.
        """
        
        raven_dir = Path(__file__).resolve().parents[1]
        
        if lc_dir  == 'default':
            self.lc_dir = raven_dir / 'Lightcurves'
        else:
            self.lc_dir = Path(lc_dir)
           
        self.dir_style = dir_style
            
        if infile_type == 'exofop':
            self.data = cutils.load_exofop_toi(infile, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'archive':
            self.data = cutils.load_archive_toi(infile, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'recovery':
            self.data = cutils.load_recovery(infile, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'default':
            self.data = cutils.load_default(infile, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'dataframe':
            self.data = cutils.process_dataframe_input(infile.copy(), per_lim=per_lim, depth_lim=depth_lim)
        else:
            raise ValueError('Infile type must be set as one of: default/archive/exofop')
        
        sectors =  pd.read_csv(raven_dir / f'Input/{sector_file}').set_index('ticid')
        sectors = sectors.loc[sectors.index.intersection(self.data.index.unique('ticid'))]
        sectors['sector'] = sectors['sector'].apply(lambda x: [int(s) for s in x.split(',')[:-1]])
        
        self.data = self.data.join(sectors, how='inner')
        self.data.rename(columns={'sector':'sectors'}, inplace=True)
        
        # Data containers    
        self.sources = {} # Source objects
        self.sector_data = None # Per sector transit data
        self.centroid = None # Observed centroid offsets
        self.probabilities = pd.DataFrame() # Positional Probabilities
        self.assessment = pd.DataFrame() # Assessment for all nearby sources
        self.features = pd.DataFrame() # ML features
        self.som_array = np.array([])
        self.priors = pd.DataFrame()
        self.candidate_classifications = pd.DataFrame()
        self.validation_results = pd.DataFrame()

        # Status variables for the different steps of the pipeline
        self.find_stars = False
        self.sectordata = False
        self.centroiddata = False 
        self.flux_fractions = False
        self.estimate_depths = False
        self.possible_sources = False
        self.probabilities_generated = False
        
        # Number of processors for multiprocessed workloads
        self.multiprocessing = multiprocessing
        
        # Indicate whether the output of the pipeline should be saved. True by default.
        self.save_output = save_output
        
        # Set a suffix that will be attached to the names of the files generated by the different 
        # process of the pipeline. If not user specified, use the date and time of class creation.
        if save_output is not False:   
            if save_suffix:
                self.save_suffix = save_suffix
            else:
                self.save_suffix = datetime.today().strftime('%d%m%Y_%H%M%S')
        else:
            self.save_suffix = ''
        
        # Define and create output directory if it does not exist       
        self.output = Path(__file__).resolve().parents[1] / 'Output'
        self.output.mkdir(exist_ok=True)
        
        # Suffix to load output from previous run. Affects all process of the pipeline.
        self.load_suffix = load_suffix
        
        self.plot_centroid = plot_centroid
        # Define and create the directory to save the centroid plots. 
        # Individual folders for each target will be created later within this directory
        if self.plot_centroid:
            outfile = self.output / 'Plots'
            outfile.mkdir(exist_ok=True)
            
    def generate_sources(self, infile=None, rerun=False):
        """
        Retrieve stellar characteristics from the TIC for each target star 
        and generate the source object data-containeres that will be used to run the pipeline. 
        Identify nearby TIC stars for each candidate, up to 8.5 ΔTmag.
        
        Parameters:
        infile - Overwrite the load suffix specified in class creation to load a user specified file.
        rerun - force the nearby star indentification process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if sources already exist
        """
        # Get the unique ticids. Multiple candidates on one source will result to only one source created.
        targets = self.data.index.unique('ticid')

        # Load in data from file, if provided at class initialisation
        preload = False
        if (self.load_suffix or infile) and not rerun and not self.find_stars:
            if infile:
                infile = Path(infile)
            else:
                infile = self.output / f'sources_{self.load_suffix}.pkl'
            print('Loading from ' + str(infile))
            try:
                with open(infile, 'rb') as f:
                    self.sources = pickle.load(f)

                # Compare loaded sources ids with the ids of the targets in the data to find any that might be missing
                targets = np.setdiff1d(targets, list(self.sources.keys()), assume_unique=True)
                preload = True
            except Exception as e:
                # Handle failure to load data
                print(e)
                print('Error loading saved sources, recreating...')
                pass
        
        # If the function is called multiple times during execution, this allows for reusing the existing sources.
        # Usefull in case of disconnections when retrieving data from MAST
        if not preload and not rerun and len(self.sources.keys()) > 0:
            targets = np.setdiff1d(targets, list(self.sources.keys()), assume_unique=True)
            print('Existing sources found and are being reused')
            
        existing_sources = self.sources.keys()
                
        if len(targets) > 0:
            print(f'Retrieving data from MAST for {len(targets)} targets')
            source_fail = []
            duplicate = []
            artifact = []
            non_gaia = []
            non_gaia_sources = 0
            
            # Run the identification process
            if self.multiprocessing > 1 and len(targets) > 10:
                # Multithread if requested and there are enough targets to be worth it
                # Set the number of mutlithreading workers. Maximum of 20 to not overload MAST with requests
                workers = np.min((self.multiprocessing*4, 20)) 
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    try:
                        futures = {ex.submit(self.find_TIC_stars, targetid): targetid for targetid in targets}
                        
                        # Retrieve results as they are returned
                        for future in as_completed(futures):
                            targetid = futures[future] 
                            try:
                                source_obj, fail, num_non_gaia = future.result()
                                if source_obj and fail is None:
                                    # Store the source object in the sources class dictionary if it was successfully created 
                                    self.sources[targetid] = source_obj
                                    non_gaia_sources += num_non_gaia
                                else:
                                    # Handle the failure cases 
                                    if fail == 'DUPLICATE':
                                        print(f'{targetid} skipped due to being flagged as a duplicate')
                                        duplicate.append(targetid)
                                    elif fail == 'ARTIFACT':
                                        print(f'{targetid} skipped due to being flagged as an artifact')
                                        artifact.append(targetid)
                                    elif fail == 'NonGaia':
                                        self.sources[targetid] = source_obj
                                        non_gaia_sources += num_non_gaia
                                        print(f'{targetid} flagged as it has no Gaia ID')
                                        non_gaia.append(targetid)
                                    else:
                                        print(f'{targetid} source failed to be created.')
                                        source_fail.append(targetid)
                                    
                            except Exception as e:
                                # Handle uncaught exceptions
                                print(f'Source not created for {targetid} due to exception: {e}')
                                source_fail.append(targetid)
           
                    except KeyboardInterrupt:
                        # Attempt to shutdown the multithreaded workload if interrupted
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')
            else:
                # Single threaded workload
                for targetid in targets:
                    source_obj, fail, num_non_gaia = self.find_TIC_stars(targetid)
                    if source_obj and fail is None:
                        # Store the successfully created sources
                        self.sources[targetid] = source_obj
                        non_gaia_sources += num_non_gaia
                    else:
                        # Handle the failure cases 
                        if fail == 'DUPLICATE':
                            print(f'{targetid} skipped due to being flagged as a duplicate')
                            duplicate.append(targetid)
                        elif fail == 'ARTIFACT':
                            print(f'{targetid} skipped due to being flagged as an artifact')
                            artifact.append(targetid)
                        elif fail == 'NonGaia':
                            self.sources[targetid] = source_obj
                            non_gaia_sources += num_non_gaia
                            print(f'{targetid} flagged as it has no Gaia ID')
                            non_gaia.append(targetid)
                        else:
                            print(f'{targetid} source failed to be created.')
                            source_fail.append(targetid)
            
            # Handle duplicate target sources
            
            self.data.drop(duplicate, axis=0, level=0, inplace=True)
            # Remove artifact sources
            self.data.drop(artifact, axis=0, level=0, inplace=True)
            # Output completion log
            print('Source identification and object creation completed.')
            if len(duplicate) > 0:
                print(f'{len(duplicate)} duplicate target source(s) removed:', duplicate)
            if len(artifact) > 0:
                print(f'{len(artifact)} artifact target source(s) removed:', artifact)
            if len(non_gaia) > 0:
                print(f'{len(non_gaia)} non Gaia target source(s) flagged:', non_gaia)
            
            print(f'{non_gaia_sources} non Gaia source(s) identified.')
            
            if len(source_fail) > 0:   
                print(f'{len(source_fail)} source(s) failed to be created:', source_fail)
            else:
                # Mark the process as completed
                self.find_stars = True
        else:
            self.find_stars = True
        
        # Save output to be reused if specified when class was initialised
        if self.save_output and self.sources.keys() != existing_sources:
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
                
    @staticmethod            
    def find_TIC_stars(targetid):
        '''
        Function for retrieving the stellar characteristics of target stars and their nearby identified sources.
        Separated to aid mutlti-threaded runs.
        
        Parameters:
        targetid - the TIC id of the target star
        '''
        source_obj = None 
        fail = None # Handle for failure cases
        non_gaia = 0 # Count of tic sources not in Gaia
        s_rad = 168 * u.arcsec  # 8 TESS pixels
        
        # Retrieve the data for the target and its nearby sources
        sources_data = cutils.query_tic_by_id(targetid, s_rad)

        # Handle failure to find target
        if sources_data is None:
            return source_obj, fail, non_gaia
        
        # Select the target data
        target_data = sources_data.query(f'ticid == {targetid}').iloc[0]
        # Check if target is flagged as duplicate or artifact
        if target_data['disposition'] == 'ARTIFACT' or target_data['disposition'] == 'DUPLICATE':
            fail = target_data['disposition']
            return source_obj, fail, non_gaia
        if pd.isna(target_data['Gaia_id']):
            no_gaia_id_flag = 1
            fail = 'NonGaia'
            gaia_id = None
        else:
            gaia_id = int(target_data['Gaia_id'])
            no_gaia_id_flag = 0
        
        # Remove artifact or duplicate nearby sources
        sources_data.drop(sources_data.query('disposition == "DUPLICATE" or disposition == "ARTIFACT"').index, inplace=True)
        
        # Identify and record the number of Non Gaia data sources
        idx_non_gaia = pd.isna(sources_data['Gaia_id'])
        non_gaia = len(sources_data[idx_non_gaia])
        
        # Set the proper motion of non Gaia sources to 0
        sources_data.loc[idx_non_gaia, 'pmra'] = 0
        sources_data.loc[idx_non_gaia, 'pmdec'] = 0
        
        # Set nan proper motion to 0, to ease subsequent calculations
        idx_na = pd.isna(sources_data[['pmra', 'pmdec']]).any(axis=1)
        sources_data.loc[idx_na, 'pmra'] = 0
        sources_data.loc[idx_na, 'pmdec'] = 0
                                
        # Create a source object for the target
        source_obj = Source(tic = targetid,
                            coords = (target_data['ra'], target_data['dec']),
                            pm = (float(target_data['pmra']), float(target_data['pmdec'])), 
                            dist = target_data['dist'],
                            Tmag=target_data['Tmag'],
                            Gmag=target_data['Gmag'],
                            radius = target_data['rad'],
                            mass = target_data['mass'],
                            teff = target_data['Teff'])

        # Retrieve the Gaia DR3 data for the target
        dr3 = cutils.query_Gaia_by_coords(targetid, target_data['ra'], target_data['dec'], 1*u.arcsec)
        if len(dr3) > 0:
            try:
                gaia_data = dr3.loc[gaia_id]
                dr3_flag = 0
            except KeyError:
                if len(dr3) == 1:
                    gaia_data = dr3.iloc[0]
                    dr3_flag = 'dr2_dr3_mis'
                else:
                    dr3_flag = 'dr3_mis_fail'
                    gaia_data = None
        else:
            dr3_flag = 'no_dr3'
            gaia_data = None
                
        if gaia_data is not None:
            # Store the RUWE value
            source_obj.ruwe = gaia_data['RUWE']
            # The Gaia BP-RP
            source_obj.bp_rp = gaia_data['BP-RP']
            # Set the distance based on the DR3 parallax, to match the simulated data.
            # The TIC distance set above is used in case the dr3 query fails.
            source_obj.dist = 1000/gaia_data['Plx']
            # Update the Gmag
            source_obj.Gmag = gaia_data['Gmag']

        # Set the no gaia id flag
        source_obj.no_gaia_id = no_gaia_id_flag
        # Store the number of non gaia nearby sources
        source_obj.non_gaia_sources = non_gaia
        # Set the dr3 flag for position and proper motion
        source_obj.Gaia_dr3_flag = dr3_flag
        
        # If the stellar radius or effective temperature were not provided in the TIC,
        # attempt to retrieve them directly from the Gaia DR2 dataset.
        # if np.isnan(source_obj.radius) | np.isnan(source_obj.teff):
        #     gaia_data = cutils.GAIA_byTICID(targetid)
        #     if gaia_data:
        #         if np.isnan(source_obj.radius):
        #             source_obj.radius = gaia_data['radius_val'][0]
        #             source_obj.Gaia_rad_flag = 1
        #         if np.isnan(source_obj.teff):
        #             source_obj.teff = gaia_data['teff_val'][0]
        #             source_obj.Gaia_teff_flag = 1


        # Remove faint sources with dTmag greater than 10
        tmag_limit = source_obj.Tmag + 10
        sources_data.query(f'Tmag <= {tmag_limit}', inplace=True)
        sources_data.set_index('ticid', inplace=True)
            
        # Use the Tmag to calculate the expected flux counts observed by each source
        sources_data['Flux'] = 15000 * 10 ** (-0.4 * (sources_data['Tmag'] - 10))
            
        # Ensure that the target is always first in the dataframe
        sources_data['order'] = np.arange(len(sources_data))  + 1
        sources_data.loc[targetid, 'order'] = 0
        sources_data.sort_values('order', inplace=True)
        sources_data.drop('order', axis=1, inplace=True)
            
        # Store the nearby data into the source object
        source_obj.nearby_data = sources_data
        
        return source_obj, fail, non_gaia
    
    def check_target_lcs(self, donwload_missing=False):
        self.download_missing = donwload_missing
        
        ticids = self.data.index.unique('ticid')
        num_targets = len(ticids)
        
        if self.multiprocessing > 1 and num_targets > 5:
            if self.multiprocessing > num_targets:
                workers = num_targets 
            else:
                workers = self.multiprocessing
            factor = 10
            while num_targets < factor*workers:
                factor -= 1
                if factor == 1:
                    break
            chunksize = int(np.ceil(num_targets/(factor*workers)))
            
            with ProcessPoolExecutor(workers) as ex:
                sectors = list(ex.map(self.check_lcs, ticids, chunksize=chunksize))
        else:
            sectors = []
            for ticid in ticids:
                sectors.append(self.check_lcs(ticid))

        df = pd.DataFrame(sectors, columns=['ticid', 'sectors']).set_index('ticid')
        
        self.data.drop('sectors', axis=1, inplace=True)
        
        self.data = self.data.join(df, how='inner')
    
    def check_lcs(self, ticid):
        try:
            sectors = self.data.loc[[ticid]].iloc[0]['sectors']
            found_sectors = []
            for sec in sectors:
                lc_file = cutils.lc_filepath(self.lc_dir, self.dir_style, ticid, sec)
                
                if lc_file.exists():
                    found_sectors.append(sec)
                elif self.download_missing:
                    result = cutils.download_spoc_lc(ticid, sec, lc_file.parent)
                    if result:
                        found_sectors.append(sec)
                else:
                    print(ticid, sec, 'Missing')
            return (ticid, found_sectors)
        except Exception:
            return (ticid, [])

    def generate_per_sector_data(self, skip_per_correction=False, infile=None, rerun=False):
        """
        Determine the per sector depth, transit duration and the duration without ingress/egrees.
        
        Parameters:
        infile - Overwrite the load suffix specified in class creation to load a user specified file.
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if sector_data already exist
        """
        
        # Check if the stars and the nearbies have been identified and the sources dictionary has been populated. 
        # Raise an error if not.
        if not self.find_stars:
            raise ValueError('Run find_tic_stars first')
                    
        self.skip_per_correction = skip_per_correction
        
        # Load in data from file, if provided at class initialisation or when the function was called.
        preload = False
        if (self.load_suffix or infile) and not rerun and not self.sectordata:
            if infile:
                infile = Path(infile)
            else:
                infile = self.output / f'sectordata_{self.load_suffix}.csv'
            print('Loading from ' + str(infile))
            try:
                data_preloaded = pd.read_csv(infile).set_index(['ticid', 'candidate', 'sector'])
                preload = True
            except Exception as e:
                # Handle failure to load existing data
                print(e)
                print('Error loading existing sector data, recreating...')
        
        # Create empty multi-index for the ticid, candidate and sector combinations. 
        indx = pd.MultiIndex(names=['ticid', 'candidate', 'sector'], levels=[[],[],[]], codes=[[],[],[]])
        for ticid in self.data.index.unique('ticid'):
            cndts = self.data.loc[ticid].index
            sectors = self.data.loc[ticid].sectors.iloc[0]
            indx = indx.append(pd.MultiIndex.from_product([[ticid], cndts, sectors], names=['ticid','candidate','sector']))
        
        # Create new dataframe with the multi-index constructed above, data initialised with 0
        new_df = pd.DataFrame(data=0, index=indx, columns=['t0', 'per', 'new_per', 'new_t0', 'sec_tdur', 'sec_tdur23', 'sec_depth', 'sec_flux', 'sec_time'])
                
        if not rerun:
            if not preload:
                # Check if sector_data already exists and update the newly constructed dataframe with existing values
                if self.sector_data is not None:
                    new_df = pd.concat([new_df, self.sector_data])
                    new_df = new_df[~new_df.index.duplicated(keep='last')]   
            else:           
                # Update the dataframe with the preloaded values
                new_df = pd.concat([new_df, data_preloaded])
                new_df = new_df[~new_df.index.duplicated(keep='last')]
    
        self.sector_data = new_df
        
        self.sector_data = self.sector_data.loc[self.data.index.unique('ticid')]

        # Find the entries which are still zero. Those will need to be filled.
        to_fill = self.sector_data.query('t0 == 0')
        ticids = to_fill.index.unique('ticid')
        num_targets = len(ticids)

        if num_targets > 0:
            filled_lst = []
            print(f'Running sector_data update for {num_targets} targets')
            if self.multiprocessing > 1 and num_targets > 5:
                # Run multi-processed if requested at class initialisation and there are enough targets to be worth it
                if num_targets < self.multiprocessing:
                    # If there are less targets than the specified number of cores, set the number of workers accordingly
                    workers = num_targets
                else:
                    workers = self.multiprocessing
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    try:
                        # Split the data into chunks based on the number of workers, to aid multiprocessing performance
                        factor = 10
                        while num_targets/(factor*workers) < 8:
                            factor -= 1
                            if factor == 1:
                                break
                        
                        ticid_split = np.array_split(ticids, factor*workers)
                        #print(len(ticid_split))  
                        
                        # Run the multiprocessed job, in a chunk based approach
                        futures = {ex.submit(self.multi_target_data, ticid_group): ticid_group for ticid_group in ticid_split}
                        
                        for future in as_completed(futures):
                            # Handle the results as they are completed. 
                            try:
                                filled, fails = future.result()
                                filled_lst.append(filled)
                                if len(fails) > 0:
                                    # For individual exceptions on a target, explicitly caught and handled in the code
                                    for fail in fails:
                                        print(f'Exception {fail[1]} occur with ticid: {fail[0]}')
                            except Exception as e:
                                # For exceptions that were not caught and handled in the code, which lead to failure for the whole group of ticids
                                group = futures[future]
                                print(f'Exception {e} occur with ticid group: {group}')
                    except KeyboardInterrupt:
                        # Attempt to shutdown multi-processed work, if interrupted
                        ex.shutdown(wait=False, cancel_futures=True)
                        # Attempt to save
                        filled = pd.concat(filled_lst)        
                        self.sector_data = pd.concat([self.sector_data, filled])
                        self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]
                        raise ValueError('Keyboard interrupt')            
            else:
                # Run on a single core
                ticid_split = np.array_split(ticids, int(np.ceil(len(ticids)/10)))
                for ticid_group in ticid_split:
                    filled, fail = self.multi_target_data(ticid_group)
                    
                    filled_lst.append(filled)
                
                    for ticid, e in fail:
                        print(f'Exception {e} occur with ticid: {ticid}')
            
            # Concatenate the filled sector data for all candidates and then update the existing dataframe   
            filled = pd.concat(filled_lst)        
            self.sector_data = pd.concat([self.sector_data, filled])
            self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]
        
        self.sector_data.sort_index(inplace=True)   
        # Save data if specified when class was initialised                                      
        if self.save_output:
            outfile = self.output / f'sectordata_{self.save_suffix}.csv'
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.sector_data.to_csv(outfile)
                    
        # Mark the process as completed      
        self.sectordata = True
                
    def multi_target_data(self, ticids):
        """
        Runs the sector_data generation for a group of targets. 
        
        Parameters:
        ticids - An array of target TIC IDs
        """
        results = []
        fails = []
        
        if not self.skip_per_correction:
            try:
                functions = gbls.compile_bls()
            except ModuleNotFoundError:
                functions = None
        else:
            functions = None
        
        # Loop through the ticids to produce dataframes with ticid/canidate/sector entries
        for ticid in ticids:
            try:
                # Store the dataframes in a list
                results.append(self.per_sector_data(ticid, self.data.loc[ticid], functions))
            except Exception as e:
                # Store the failure cases and their associated exception
                fails.append((ticid, e))
                
        # Concatenate the group dataframes into one
        try:
            filled = pd.concat(results, ignore_index=True)
            filled.set_index(['ticid', 'candidate', 'sector'], inplace=True)
        except ValueError:
            # Incase that results is empty due to all targets failing
            filled = None
        
        return filled, fails
        
    def per_sector_data(self, ticid, initial_data, functions=None):
        """
        Determine per sector transit duration, non-ingress/egress duration and depth for a single candidate by fitting a trapezium model to the TESS lightcurve 
        
        Parameters:
        ticid - target TIC ID
        initial_data - dataframe entry with the target's initial data
        """

        # Retrieve the lc location, sectors and candidates from the provided data
        sectors = sorted(initial_data.iloc[0].sectors)
        candidates = initial_data.index.to_numpy()
        
        # Empty lists to store and then stitch the sector lcs into one
        fluxes = []
        times = []
        errors = []
        
        # Load and store the detrended and normalised lightcurves per sector of observation
        lcs = {}
        hdus = {}
        sec_error = []
        for sec in sectors:
            lcfile = cutils.lc_filepath(self.lc_dir, self.dir_style, ticid, sec)
            
            try:            
                lc, hdu = cutils.load_spoc_lc(lcfile, flatten=True, transitcut=True,
                                        tc_per=initial_data.per.values, tc_t0=initial_data.t0.values,
                                        tc_tdur=initial_data.tdur.values, return_hdu=True)
            except FileNotFoundError:
                continue
            
            # Set the flux MAD as the error
            lc['error'] = np.full_like(lc['flux'], futils.MAD(lc['flux']))
            sec_error.append(lc['error'][0])
            lcs[sec] = lc
            hdus[sec] = hdu
            
            fluxes.append(lc['flux'])
            times.append(lc['time'])
            errors.append(lc['error'])

        lc = {'flux':np.concatenate(fluxes),
              'time':np.concatenate(times),
              'error':np.concatenate(errors)}
        
        sector_data = [] # List to store the results produced per candidate 
        
        # Process each candidate separately
        for cndt in candidates:            
            # Create empty dataframe to store the per sector data
            cols = ['ticid', 'candidate', 'sector', 't0', 'per', 'new_per', 'sec_tdur', 'sec_tdur23', 'sec_depth', 'sec_flux', 'sec_time']
            df = pd.DataFrame(index=range(len(sectors)), columns=cols)
            # Retrieve provided t0, per, tdur, depth per candidate
            cndt_t0 = initial_data.loc[cndt, 't0']
            cndt_per = initial_data.loc[cndt, 'per']
            cndt_tdur = initial_data.loc[cndt, 'tdur']
            cndt_depth = initial_data.loc[cndt, 'depth']
            
            # Fill in the dataframe with the existing values
            df['ticid'] = ticid
            df['candidate'] = cndt
            df['sector'] = lcs.keys()
            df['t0'] = cndt_t0
            df['per'] = cndt_per
            
            # Calculate the number of observed transits based on the provided ephmeris
            num_transits = futils.observed_transits(lc['time'], cndt_t0, cndt_per, cndt_tdur)
            
            # Set whether period correction will run based on a number of conditions
            if self.skip_per_correction:
                # If generate_sector_data is run with the per correction disabled
                skip_per_cor = True
            elif functions is None:
                # If the bls function was not compiled 
                # (Possibly because of the lack of an Nvidia GPU or CUDA incompatibility)
                skip_per_cor = True
            elif len(candidates) > 1:
                # Skip correction if there is more than one candidate per target
                # to avoid transits from other candidates affecting the process
                skip_per_cor = True
            elif num_transits < 2:
                # Skip correction if there is less than 2 transits present in the data
                skip_per_cor = True
            else:
                skip_per_cor = False
            
            # Provide a provisional period for candidates without a known period (monotransits) so that they can be examined
            if cndt_per == 0 or np.isnan(cndt_per):
                cndt_per = futils.no_per(lc['time'], cndt_t0, cndt_tdur)
                if not cndt_per:
                    # If a provisional period could not be provided, set their sector data to the existing ones.
                    df['new_per'] = df['per']
                    df['new_t0'] = cndt_t0
                    df['sec_tdur'] = cndt_tdur
                    df['sec_tdur23'] = np.nan
                    df['sec_depth'] = np.nan

                    sector_data.append(df)

                    # Do not process them any further
                    continue
                
                # Skip period correction for these candidates
                skip_per_cor = True
                                                      
            if not skip_per_cor:
                time = lc['time']
                flux = lc['flux']
                
                dflux = np.full_like(flux, futils.MAD(flux))
                
                n_orbits = np.floor((time[-1]-time[0])/cndt_per)
                if n_orbits > 0:                
                    dt = cndt_tdur/(20*n_orbits)
                    dt = np.max((dt, 1e-5))

                    diff = 0.015
                    per_range = np.arange(cndt_per*(1-diff), cndt_per*(1+diff)+dt, dt)
                else:
                    per_range = np.array([cndt_per])
                
                fine_grid = 1/per_range
                fine_grid = np.flip(fine_grid)

                try:
                    new_power, sols = gbls.eebls_gpu(time, flux, dflux, fine_grid, 
                                        qmin=cndt_tdur*0.9/cndt_per, qmax=cndt_tdur*1.1/cndt_per, dlogq=0.1, 
                                        noverlap=3, functions=functions,
                                        ignore_negative_delta_sols=True, freq_batch_size=len(per_range))
                    
                    idx = np.argmax(new_power)
                    
                    new_per = np.flip(per_range)[idx]
                    
                    new_t0 = tutils.get_bls_t0(time, 1/new_per, sols[idx][1], sols[idx][0])
                    
                    new_tdur = sols[idx][0]*new_per
                    new_depth = futils.calculate_depth(time, flux, new_per, new_t0, new_tdur)[0]*1e6
                except Exception:
                    # The fine grid BLS failed. Set the transit ephemeris to the provided ones
                    new_per = cndt_per
                    new_t0 = cndt_t0
                    new_tdur = cndt_tdur
                    new_depth = cndt_depth
            else:
                new_per = cndt_per
                new_t0 = cndt_t0
                new_tdur = cndt_tdur
                new_depth = cndt_depth
                
            # Determine the per sector transit parameters using a trapezoid fit
            sec_tdur = []
            sec_tdur23 = []
            sec_depth = []
                                
            for sec in sectors:
                if new_per != cndt_per:
                    sec_lc, hdu = cutils.load_spoc_lc(filepath=None, hdu=hdus[sec], flatten=True, transitcut=True,
                                                tc_per=new_per, tc_t0=cndt_t0,
                                                tc_tdur=new_tdur, return_hdu=True)
                    sec_lc['error'] = np.full_like(sec_lc['flux'], futils.MAD(sec_lc['flux']))
                else:
                    sec_lc = lcs[sec]

                fit_tdur, fit_tdur23, fit_depth = futils.transit_params(sec_lc, new_t0, new_per, new_tdur, new_depth) 
                
                sec_tdur.append(fit_tdur)
                sec_tdur23.append(fit_tdur23)
                sec_depth.append(fit_depth)
            
            # Check if the sector depth is not nan in each sector.
            if np.isnan(sec_depth).all():
                # Replace the sector data with the existing parameters as the method has failed.
                new_per = cndt_per
                new_t0 = cndt_t0
                sec_tdur = cndt_tdur
                sec_tdur23 = np.nan
                sec_depth = cndt_depth
                
            # Set the 0 or nan periods back to their original values
            if df['per'].values[0] == 0 or np.isnan(df['per'].values[0]):
                new_per = df['per'].values[0]
            
            # Store the sector parameters in the dataframe
            df['new_per'] = new_per
            df['new_t0'] = new_t0
            df['sec_tdur'] = sec_tdur
            df['sec_tdur23'] = sec_tdur23
            df['sec_depth'] = sec_depth
            
            # Append the dataframe to the list
            sector_data.append(df)
        
        # Concatenate the candidate datraframes in one
        sector_data = pd.concat(sector_data, ignore_index=True)
        
        # Add the sector flux and spoc crowding info
        for sec in sectors:
            idx = sector_data.query(f'sector == {sec}').index
            sector_data.loc[idx, 'sec_flux'] = lcs[sec]['median']
            sector_data.loc[idx, 'sec_time'] = lcs[sec]['time'][0]
            
            hdu = hdus[sec]
            sector_data.loc[idx, 'sec_C'] = hdu[1].header['CROWDSAP']
            sector_data.loc[idx, 'sec_f'] = hdu[1].header['FLFRCSAP']
            
            # Ensure that the opened fits files are closed
            hdu.close()
            
            del hdu
                                                    
        return sector_data
                
    def generate_centroiddata(self, infile=None, rerun=False):
        """
        Calculates the centroid offset in tranist for the dataset
        
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if centroid_data already exist
        """
        # Check if the per sector data generation process has been completed before running this process, as it makes use of the sector data. 
        # Raise an error if not.
        if not self.sectordata:
            raise ValueError('Run generate_per_sector_data first')
        
        preload = False
        if (self.load_suffix or infile) and not rerun and not self.centroiddata:
            if infile:
                infile = Path(infile)
            else:
                infile = self.output / f'centroiddata_{self.load_suffix}.csv'
                
            print('Loading from ' + str(infile))
            try:
                centroid_preloaded = pd.read_csv(infile).set_index(['ticid', 'candidate', 'sector'])
                centroid_preloaded.loc[centroid_preloaded['flag'].isna(), 'flag'] = ''
                preload = True
            except:
                print('Error loading infile centroid data, recreating..')
        
        # Create empty multi-index
        indx = pd.MultiIndex(names=['ticid', 'candidate', 'sector'], levels=[[],[],[]], codes=[[],[],[]])
        for ticid in self.data.index.unique('ticid'):
            cndts = self.data.loc[ticid].index
            sectors = self.data.loc[ticid].iloc[0].sectors
            indx = indx.append(pd.MultiIndex.from_product([[ticid], cndts, sectors], names=['ticid','candidate','sector']))
        
        # Create empty dataframe with the multi-index constructed above. Initialise data entries with 0
        new_df = pd.DataFrame(data=0, index=indx, columns=['cam', 'ccd', 'X_diff', 'X_err', 'Y_diff', 'Y_err', 'flag'])
        
        # Set the flag entries to empty string
        new_df['flag'] = ''
        
        if not rerun:
            if not preload:
                # Check if sector_data already exists and update the newly constructed dataframe with existing values
                if self.centroid is not None:
                    new_df = pd.concat([new_df, self.centroid])
                    new_df = new_df[~new_df.index.duplicated(keep='last')]   
            else:           
                # Update the dataframe with the preloaded values
                new_df = pd.concat([new_df, centroid_preloaded])
                new_df = new_df[~new_df.index.duplicated(keep='last')]
        
        self.centroid = new_df
        
        # Find the entries which are still zero
        to_fill = self.centroid.query('X_diff == 0 & X_err == 0 & Y_diff == 0 & Y_err == 0')
        num_targets = len(to_fill.index.unique('ticid'))
        
        if num_targets > 0:
            print(f'Running centroid data retrieval for {num_targets} targets')
            
            filled_lst = []
            
            if self.multiprocessing > 1 and num_targets > 5:
                # Run multi-processed if requested at class initialisation and there are enough targets to be worth it
                if num_targets < self.multiprocessing:
                    # If there are less targets than the specified number of cores, set the number of workers accordingly
                    workers = num_targets
                else:
                    workers = self.multiprocessing
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    try:
                        # Split the data into chunks based on the number of workers, to aid multiprocessing performance
                        factor = 20
                        while len(to_fill.index) < 5*factor*workers:
                            factor -= 1
                            if factor == 1:
                                break
                        index_split = np.array_split(to_fill.index, factor*workers)
                        
                        # Run the mutliprocessed job, in a chunk based approach
                        # Compared to the sector_data process, the chunks include target, candidate, sector entries, not just ids
                        futures = {ex.submit(self.multi_centroid, index_group): index_group for index_group in index_split}
                        
                        for future in as_completed(futures):
                            # Handle the results as they are completed. 
                            try:
                                filled, fails = future.result()
                                filled_lst.append(filled)
                                if len(fails) > 0:
                                    # Individual exceptions, explicitly caught, handled and reported in the code
                                    for fail in fails:
                                        print(f'Exception "{fail[1]}" occur for: {fail[0]}')
                            except Exception as e:
                                # Exceptions that were not caught and handled in the code, which lead to failure for the whole chunk
                                group = futures[future]
                                print(f'Exception "{e}" occur for index group: {group}')
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')
                    
                # Concatenate the results from all processes
                filled = pd.concat(filled_lst)
                           
            else:
                # Run on a single core
                filled, fails = self.multi_centroid(to_fill.index)
                
                for fail in fails:
                    print(f'Exception {fail[1]} occur for index: {fail[0]}')
                    
            # Update the existing centroid results
            self.centroid = pd.concat([self.centroid, filled])
            self.centroid = self.centroid[~self.centroid.index.duplicated(keep='last')]   
        
        self.centroid.sort_index(inplace=True)
        # Save date if specified when class was initialised                           
        if self.save_output:
            outfile = self.output / f'centroiddata_{self.save_suffix}.csv'
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.centroid.to_csv(outfile)
        
        # Mark the process as completed         
        self.centroiddata = True
        
        
    def multi_centroid(self, indices):
        """
        Runs the centroid_data generation for a group of target/candidate/sector entries
        
        Parameters:
        indices - An array of target/candidate/sector indices
        """
        results = []
        fails = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Loop through the indices to calculate the centroid offset for the candidate on each sector
            for idx in indices:
                ticid, cndt, sec = idx
                try:
                    # Store the results in a list, to be used for the construction of a dataframe
                    results.append(self.observed_centroid_offset(ticid, cndt, sec))
                except Exception as e:
                    # Store the failure cases and their associated exception
                    fails.append((idx, e))
        
        # Construct dataframe from all individual results to return a final datframe for the whole group        
        filled = pd.DataFrame([r for r in results], columns=['ticid', 'candidate', 'sector', 'cam', 'ccd', 'X_diff', 'X_err', 'Y_diff', 'Y_err', 'flag'])
        filled.set_index(['ticid', 'candidate', 'sector'], inplace=True)
        
        return filled, fails
    
                          
    def observed_centroid_offset(self, ticid, cndt, sec):
        """
        Determine per sector centroid offset for a candidate
        
        Parameters:
        ticid - target TIC ID
        cndt - candidate number or toi id
        sec - TESS sector on which the target was observed
        """
        
        # Retrieve the per sector data [per, t0, tdur, tdu23]
        event_data = self.sector_data.loc[(ticid, cndt, sec)]
        tc_per = event_data.new_per
        tc_t0 = event_data.new_t0
        tc_tdur = event_data.sec_tdur
        tc_tdur23 = event_data.sec_tdur23

        # Sector lightcurve file
        lc_file = cutils.lc_filepath(self.lc_dir, self.dir_style, ticid, sec)
        
        # Load in the detrended, normalized and with outliers removed vertical and horizontal centroid position data. 
        # The flag specifies issues found during the loading of the data that prevent the offset calculation.
        # CAM and CCD retrieve for diagnostic purposes when displaying the results.
        time, data_X, data_Y, cent_flag, cam, ccd = cutils.load_spoc_centroid(lc_file,
                                                                                flatten=True, cut_outliers=5, trim=True,
                                                                                transitcut=False,
                                                                                tc_per=tc_per,
                                                                                tc_t0=tc_t0,
                                                                                tc_tdur=tc_tdur)
        
        if not cent_flag:
            # Calculate the centroid offsets and their associated errors by fitting a trapezium model
            X_diff, X_diff_err, Y_diff, Y_diff_err, cent_flag = cutils.centroid_fitting(ticid, cndt, sec,
                                                                                       time, data_X, data_Y,
                                                                                       tc_per, tc_t0,
                                                                                       tc_tdur, tc_tdur23, 
                                                                                       loss='huber', plot=self.plot_centroid)

            # Handle fitting failure without being flagged
            if  (np.isnan(X_diff) or np.isnan(Y_diff)) and not cent_flag:
                cent_flag = 'Nan from fitting'
        else:
            # Set offset to nan if the sector was not suitable for calculation
            X_diff, X_diff_err, Y_diff, Y_diff_err = np.nan, np.nan, np.nan, np.nan
            
        return (ticid, cndt, sec, cam, ccd, X_diff, X_diff_err, Y_diff, Y_diff_err, cent_flag)

                    
    def estimate_flux_fractions(self, rerun=False):
        """
        Determine the flux fraction contribution in the TESS aperture of the target stars and their nearby sources
        
        Parameters:
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if flux fractions were previously determined.
        """
        
        # Check if the per sector data generation process has been completed. Raise an error if not.
        if not self.sectordata:
            raise ValueError('Run generate_sector_data first')
        
        print('Estimating flux fractions')
        for ticid in self.sources.keys():
            # Retrieve the source object data container
            source_obj = self.sources[ticid]

            if source_obj.nearby_fractions is None or rerun:
                # Initialise a dataframe, with the same index as that of the nearby data, to store the fractions
                source_obj.nearby_fractions = pd.DataFrame(index=source_obj.nearby_data.index)
            
            # Retrieve the sectors    
            sectors = self.data.loc[ticid].iloc[0].sectors

            # Check if sector fractions already present in the dataframe.
            # Difference will be only if the nearby_fractions dataframe was already existing in the source_obj data.
            # Allows to run the process only for new sectors
            already_run = np.array([int(x[1:]) for x in source_obj.nearby_fractions.columns])
            sectors = np.setdiff1d(np.array(sectors), already_run)
                                     
            for sec in sectors:
                # Retrieve the lc file for the sector
                lc_file = cutils.lc_filepath(self.lc_dir, self.dir_style, ticid, sec)
                
                # Load in the pipeline aperture and centroid masks for the target pixels, the wcs and the origin location of the target pixel on the ccd
                aperture_mask, centroid_mask, wcs, origin, cam, ccd = cutils.load_spoc_masks(lc_file)
                # Store the data in the object, so that they can be reused
                source_obj.wcs[sec] = wcs
                source_obj.origin[sec] = origin
                source_obj.aperture_mask[sec] = aperture_mask
                source_obj.centroid_mask[sec] = centroid_mask
                source_obj.scc.loc[sec] = [cam, ccd]
                
                # Create temporary copy of the nearby data
                temp_data = source_obj.nearby_data.copy()
                
                # Compare the estimated and observed flux of the target in the sector to correct for discrepancies 
                estimated_flux = temp_data.loc[ticid, 'Flux']
                dummy_cndt = self.data.loc[ticid].index[0]
                observed_flux = self.sector_data.loc[(ticid, dummy_cndt, sec), 'sec_flux']
                cor_factor = observed_flux/estimated_flux
                temp_data['Flux'] *= cor_factor
                
                # Store the correction factor for later use
                source_obj.cor_factor[sec] = cor_factor
                
                # Account for proper motion
                obs_time = self.sector_data.loc[(ticid, dummy_cndt, sec), 'sec_time']
                cor_ra, cor_dec = cutils.gaia_pm_corr(obs_time, temp_data['ra'].values, temp_data['dec'].values, temp_data['pmra'].values, temp_data['pmdec'].values)
                temp_data['ra'] = cor_ra
                temp_data['dec'] = cor_dec
                
                # Use the retrieved WCS to convert the ra and dec of the nearby sources into pixel locations
                temp_data['x'], temp_data['y'] = wcs.all_world2pix(temp_data.ra, temp_data.dec, 0)
                
                # Test if target position from wcs is correctly in the aperture. Catch wcs errors
                in_ap = cutils.test_target_aperture(temp_data.loc[ticid].x, temp_data.loc[ticid].y, aperture_mask)
                
                if in_ap:
                    # Calculate the flux fractions and the total flux in aperture, by modelling the observation using the TESS PRF
                    prfs = cutils.sources_prf(sec, cam, ccd, origin, temp_data['x'].values, temp_data['y'].values, aperture_mask.shape)
                    fractions, total_flux = cutils.flux_fraction_in_ap(temp_data.Flux.values, aperture_mask, prfs)
                else:
                    # Set the fractions to nan, to effectively ignore this sector
                    prfs = None
                    fractions = np.zeros(len(temp_data))
                    fractions[:] = np.nan
                    
                    total_flux = np.nan
                
                # Store the model prfs, fractions, the modeled total flux and the Tmag equivalent of the flux in the aperture
                source_obj.prfs[sec] = prfs
                source_obj.nearby_fractions[f'S{sec}'] = fractions 
                source_obj.totalflux[sec] = total_flux
                source_obj.totalmag_equivalent[sec] = 10 - 2.5*np.log10(total_flux/15000)
                     
        # Save output to be reused
        if self.save_output:
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
        
        # Mark the process as completed       
        self.flux_fractions = True


    def estimate_nearby_depths(self, rerun=False):
        """
        Determine the flux fraction contribution in the TESS aperture of the target stars and their nearby sources
        
        Parameters:
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if nearby depths were previously determined.
        """
         # Check if flux fractions have been computed. Raise an error if not.
        if not self.flux_fractions:
            raise ValueError('Run estimate_flux_fractions first')
        
        print('Estimating depth of event on nearby stars')
        for ticid in self.sources.keys():
            # Retrieve the target data and source object
            target_data = self.data.loc[ticid]
            source_obj = self.sources[ticid]
            
            candidates = target_data.index.values
            
            # Construct a dataframe to store the depth of the event if it would occur in each of the nearby sources,
            # based on the depth of the event detected on the target on each sector. Initialised with nan.
            indx = pd.MultiIndex.from_product([source_obj.nearby_data.index.values, candidates], names=['ticid', 'candidate'])
            cols = [f'S{sec}' for sec in target_data.iloc[0].sectors]
            nearby_depths = pd.DataFrame(data=np.nan, index=indx, columns=cols)
            
            if not rerun:
                # Merged the new dataframe with the existing on, dropping the Mean depth column
                try:
                    nearby_depths = pd.concat([nearby_depths, source_obj.nearby_depths.drop('Mean', axis=1)])
                    nearby_depths = nearby_depths[~nearby_depths.index.duplicated(keep='last')]
                except KeyError:
                    pass
            
            # Store the new dataframe on the object
            source_obj.nearby_depths = nearby_depths

            # Fill in the depths per candidate
            for cndt in candidates:
                # Retrieve the existing candidate depth data
                sub_depths = source_obj.nearby_depths.query(f'candidate == {cndt}')
                # Identify the sector columns which are still not filled, i.e. all entries are null
                null_sectors = sub_depths.loc[:, sub_depths.isnull().all()].columns

                # Per sector
                for sec in null_sectors:
                    # Convert the 'S{}' column name to int
                    sec_num = int(sec[1:])

                    # Retrieve the candidate depth on target from the per-sector data
                    cndt_sec_depth = self.sector_data.loc[(ticid, cndt, sec_num), 'sec_depth']
                    
                    # If the depth was less than 50ppm, set it to nan, to effectively skip the sector 
                    if cndt_sec_depth < 50:
                        cndt_sec_depth = np.nan

                    # Retrieve the target and nearby flux fractions
                    f_target = source_obj.nearby_fractions.loc[ticid, sec]
                    f_nearby = source_obj.nearby_fractions[sec].values
                    
                    # Calculate the implied depths for the nearby sources and store them
                    depths = cutils.nearby_depth(cndt_sec_depth, f_target, f_nearby)
                    source_obj.nearby_depths.loc[source_obj.nearby_depths.index.get_level_values('candidate') == cndt, sec] = depths
            
            # Calculate the mean eclipse depth for the event based on all sectors, ignoring sectors with 0 depth
            source_obj.nearby_depths['Mean'] = source_obj.nearby_depths.replace(0, np.nan).mean(axis=1)

            # For nearby sources where the flux fraction in aperture was 0 for all sectors, set the mean to 0.
            zero_idx = source_obj.nearby_fractions[(source_obj.nearby_fractions == 0).all(axis=1)].index
            source_obj.nearby_depths.loc[zero_idx, 'Mean'] = 0
            
        # Mark the process as completed   
        self.estimate_depths = True
    
          
    def generate_probabilities(self, max_eclipsedepth=1.0, prob_thresh=1e-4, rerun=False):
        """
        Generates positional probabilities for the target and nearby sources and performs an assessment of the suitability for each to be the true host of the event.
        
        Parameters:
        max_eclipsedepth - The maximum depth allowed for an eclipse on a source to be considered valid
        max_transitdepth = The maximum depth allowed for a transit on a source to be considered as a possible planet candidate
        prob_thresh - The minimum probability for a source to be considered as a possible alternative source of the detected event
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if probabilities were previously generated.
        """
        # Check if centroid data have been generated. Raise an error if not.
        if not self.centroiddata:
            raise ValueError('Run generate_centroiddata first!')
        
        # Check if nearby depths have been determined. Raise an error if not.
        if not self.estimate_depths:
            raise ValueError('Run nearby_flux_fractions first!')
        
        print(f'Generating Positional Probabilities')
        
        if rerun:
            self.probabilities = pd.DataFrame() # Positional Probabilities
            self.assessment = pd.DataFrame() # Assessment for all nearby sources
        
        for targetid in self.data.index.unique('ticid'):
            # Retrieve the target data and source object
            target_data = self.data.loc[targetid]
            source_obj = self.sources[targetid]

            sectors = target_data.iloc[0].sectors
            
            if not rerun:
                # Run only for sectors not processed yet
                sectors_out = np.setdiff1d(sectors, source_obj.cent_out.index)
            else:
                sectors_out = sectors
            
            for sec in sectors_out:
                # Retrieve the model prfs stored before
                prfs = source_obj.prfs[sec]
                if prfs is not None:
                    temp_data = source_obj.nearby_data[['Flux']].copy()
                    cor_factor = source_obj.cor_factor[sec]
                    temp_data['Flux'] *= cor_factor
                    
                    centroid_mask = source_obj.centroid_mask[sec]
                    
                    try:
                        cent_x, cent_y = cutils.model_centroid(centroid_mask, temp_data.Flux.values, prfs)
                    except Exception as e:
                        print(targetid, e)
                        cent_x, cent_y = np.nan, np.nan
                else:
                    cent_x, cent_y = np.nan, np.nan
                                    
                # Store the out of transit centroid for each sector
                source_obj.cent_out.loc[sec] = cent_x, cent_y

            # Construct a dataframe to store the model centroid in transit per sector for the event on the target and on the nearby sources
            cent_in = pd.DataFrame(data=0.0, 
                                   index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index, sectors], names=['ticid', 'candidate', 'sector']), 
                                   columns=['X', 'Y'])
            if not rerun:
                # Merged with existing in transit model centroids
                cent_in = pd.concat([cent_in, source_obj.cent_in])
                cent_in = cent_in[~cent_in.index.duplicated(keep='last')]
                
            source_obj.cent_in = cent_in
            
            # Determine which entries still need to be processed                                                
            to_fill = source_obj.cent_in.loc[(cent_in == 0).all(axis=1)]
            if len(to_fill) > 0:
                for sec in to_fill.index.unique('sector'):
                    # Check if out of transit centroid for the sector is nan
                    if source_obj.cent_out.loc[sec].isna().any():
                        source_obj.cent_in.loc[cent_in.query(f'sector == {sec}').index] = np.nan
                        continue
                        
                    # Retrieve the sector information
                    temp_data = source_obj.nearby_data[['Flux']].copy()
                    cor_factor = source_obj.cor_factor[sec]
                    temp_data['Flux'] *= cor_factor
                    centroid_mask = source_obj.centroid_mask[sec]
                    prfs = source_obj.prfs[sec]
                    
                    # Process each candidate individually
                    for cndt in to_fill.query(f'sector == {sec}').index.unique('candidate'):
                        # Check if the observed centroid for the candidate is nan. No need to model centroid then.
                        if self.centroid.loc[targetid, cndt, sec].isna().any():
                            source_obj.cent_in.loc[cent_in.query(f'candidate == {cndt} & sector == {sec}').index] = np.nan
                            continue
                                                                    
                        # Check if the candidate depth for the sector is nan
                        cndt_depth = source_obj.nearby_depths.loc[(targetid, cndt), f'S{sec}']
                        if np.isnan(cndt_depth):
                            source_obj.cent_in.loc[cent_in.query(f'candidate == {cndt} & sector == {sec}').index] = np.nan
                            continue
                        
                        # Determine the model in transit centroid for the target and the nearby sources
                        for ticid in source_obj.nearby_data.index.unique('ticid'):
                            # Retrieve the mean depth for the source
                            depth = source_obj.nearby_depths.loc[(ticid, cndt), f'Mean']*1e-6
                            obj_type = source_obj.nearby_data.loc[ticid, 'objType']

                            # Check depth suitability
                            if depth == 0.0 or np.isnan(depth):
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            elif depth*0.9 > max_eclipsedepth:
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            elif obj_type != 'STAR':
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            else:
                                # Retrieve the fluxes
                                fluxes = temp_data['Flux'].copy()
                                
                                # Scale the flux of the source by the depth
                                depth_scale = np.min((depth, 1.0))
                                fluxes.loc[ticid] *= (1-depth_scale) 
                                
                                # Determine in transit model centroid
                                X, Y = cutils.model_centroid(centroid_mask, fluxes.values, prfs)
                                
                                # Store the model centroids
                                source_obj.cent_in.loc[ticid, cndt, sec] = [X, Y]

                # Determine the model centroid offset and the error for all candidates and sources
                model_offset = pd.DataFrame()
                model_offset['X_diff'] = source_obj.cent_in['X'] - source_obj.cent_out['X']
                model_offset['Y_diff'] = source_obj.cent_in['Y'] - source_obj.cent_out['Y']
                model_offset['X_err'] = model_offset['X_diff']*0.1
                model_offset['Y_err'] = model_offset['Y_diff']*0.1
                
                # Store the model centroid offset and errors                
                source_obj.model_centroid = model_offset[['X_diff', 'X_err', 'Y_diff', 'Y_err']]
                
                # Add the observed centroid offset to the dataframe, per sector, to ease subsequent calculations
                model_offset = model_offset.join(self.centroid.loc[targetid], on=['candidate', 'sector'], rsuffix='_obs')
                
                # Probabilistically compare the observed and model centroid offsets
                model_offset['Probability'] = model_offset.apply(lambda x: cutils.calc_centroid_probability(x['X_diff_obs'], x['Y_diff_obs'], 
                                                                                                            x['X_err_obs'], x['Y_err_obs'], 
                                                                                                            x['X_diff'], x['Y_diff'], 
                                                                                                            x['X_err'], x['Y_err']), axis=1)
                
                # Compute the sum of the probabilities for all candidates per sector
                model_offset = model_offset.join(model_offset.groupby(['candidate', 'sector']).agg(Prob_Sum=('Probability', 'sum')), on=['candidate', 'sector'])
                                
                # Compute the normalised probability for each candidate per sector
                model_offset['Norm_Probability'] = model_offset['Probability'] / model_offset['Prob_Sum']
                
                # Store the probability, probability sum and normalised probability per sector/candidate/source 
                source_obj.model_prob = model_offset[['Probability', 'Prob_Sum', 'Norm_Probability']]
                
                # Construct a dataframe for the probababilities to be reported
                prob_centroid = pd.DataFrame(index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index], names=['ticid', 'candidate']))
                
                # Calculate the Max, Mean and Median un-normalised probability for each source
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MaxProb=('Probability','max')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MeanProb=('Probability','mean')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MedianProb=('Probability','median')), on=['ticid', 'candidate'])
                
                # Calculate the Max, Mean and Median normalised probability for each source
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MaxNormProb=('Norm_Probability','max')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MeanNormProb=('Norm_Probability','mean')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MedianNormProb=('Norm_Probability','median')), on=['ticid', 'candidate'])
                
                # Calculate the sum for the median normalised probability
                prob_centroid = prob_centroid.join(prob_centroid.groupby(['candidate']).agg(NormMedianSum=('MedianNormProb','sum')), on=['candidate'])

                # Calculate the final positional probability
                prob_centroid['PositionalProb'] = prob_centroid['MedianNormProb'] / prob_centroid['NormMedianSum']
                
                # Drop the probability sum
                prob_centroid.drop(['NormMedianSum'], axis=1, inplace=True)
                                            
                # Store probability centroid to the source
                source_obj.prob_centroid = prob_centroid
                            
                # Assess the suitability of each nearby as the host of the event
                nearby_assessment = pd.DataFrame(index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index], names=['ticid', 'candidate']))
                nearby_assessment['Possible'] = True
                nearby_assessment['Rejection Reason'] = '' 
            
                # Check for zero depth and hence zero flux in aperture
                nearby_assessment.loc[source_obj.nearby_depths['Mean'] == 0] = False, 'Zero flux in aperture'
                # Check for nan Mean depth
                nearby_assessment.loc[source_obj.nearby_depths['Mean'].isna()] = False, 'Nan depth'
                # Check for depth above max eclipse depth, with some room for error
                nearby_assessment[source_obj.nearby_depths['Mean']*1e-6*0.9 > max_eclipsedepth] = False, 'Eclipse depth above max'
                            
                # For the rest check the centroid probability
                mask = nearby_assessment['Possible'] == True
                # Below threshold
                nearby_assessment.loc[source_obj.prob_centroid[mask].query(f'PositionalProb < {prob_thresh}').index] = False, 'Centroid probability below threshold'
                # Nan
                nearby_assessment.loc[source_obj.prob_centroid[mask].query(f'PositionalProb != PositionalProb').index] = False, 'Nan centroid probability'
                
                # Then check suitability as a Nearby HEB
                nearby_assessment['Possible_NHEB'] = False
                # Check if it's a possible host of the eclipse
                mask1 = nearby_assessment['Possible'] == True
                # Check if the depth is consistent with a HEB (typically ~35% but allow up to 50%)
                mask2 = source_obj.nearby_depths['Mean']*1e-6 <= 0.5
                # Combine the two boolean conditions and set possible NHEBs to True
                nearby_assessment.loc[mask1 * mask2, 'Possible_NHEB'] = True
                # Set the target star as False
                nearby_assessment.loc[targetid, 'Possible_NHEB'] = False 
                
                # Finally check suitability as a Nearby Transiting Planet
                nearby_assessment['Possible_NTP'] = False
                # Check if it's a possible host of the eclipse
                mask1 = nearby_assessment['Possible'] == True
                # Check if the depth is consistent with a panet (allow up to 10%)
                mask2 = source_obj.nearby_depths['Mean']*1e-6 <= 0.1
                # Combine the two boolean conditions and set possible NTP to True
                nearby_assessment.loc[mask1 * mask2, 'Possible_NTP'] = True 
                # Set the target star as False
                nearby_assessment.loc[targetid, 'Possible_NTP'] = False
                                
                # Store assessment in source_obj
                source_obj.nearby_assessment = nearby_assessment
                
            cent_prob = source_obj.prob_centroid.loc[source_obj.nearby_assessment.query(f'Possible == True or ticid == {targetid}').index].copy()
            try:
                disp = target_data['tfop_disp']
                if pd.isna(disp).any():
                    disp = target_data['tess_disp']
                disp.name = 'disp'
            except KeyError:
                disp = pd.DataFrame(data= ['TCE'] * len(target_data.index), index=target_data.index, columns=['disp'])
                
            cent_prob = cent_prob.join(disp)
            cent_prob.loc[cent_prob.query(f'ticid != {targetid}').index, 'disp'] = 'NFP'
            cent_prob['target'] = targetid
            cent_prob.reset_index(inplace=True)
            cent_prob.rename(columns={'ticid':'source'}, inplace=True)
            cent_prob.set_index(['target', 'candidate', 'source'], inplace=True)
            
            self.probabilities = pd.concat([self.probabilities, cent_prob])
            
            nb_assessment = source_obj.nearby_assessment.copy()
            nb_assessment['target'] = targetid
            nb_assessment.reset_index(inplace=True)
            nb_assessment.rename(columns={'ticid':'source'}, inplace=True)
            nb_assessment.set_index(['target', 'candidate', 'source'], inplace=True)
            
            self.assessment = pd.concat([self.assessment, nb_assessment])
        
        self.probabilities = self.probabilities[~self.probabilities.index.duplicated(keep='last')]    
        self.probabilities.sort_values(['target', 'candidate', 'PositionalProb'], ascending=[1,1,0], inplace=True)
        
        self.assessment = self.assessment[~self.assessment.index.duplicated(keep='last')]  
        self.assessment.sort_values('Possible', ascending=False)
        
        print('Positional Probabilities generated!')
        
        self.probabilities.rename(columns={'PositionalProb':'PositionalProbability',
                                           'disp':'Disposition'}, inplace=True)
                 
        if self.save_output == 'full':  
            # Output the probabilities
            outfile = self.output / f'Probabilities_{self.save_suffix}.csv'
            self.probabilities.to_csv(outfile)
            
            # Output the assessment
            outfile = self.output / f'Assessment_{self.save_suffix}.csv'
            self.assessment.to_csv(outfile)
            
            # Save the sources to be reused
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
        elif self.save_output is True:
            # Output just the Positional Probabilities
            outfile = self.output / f'Probabilities_{self.save_suffix}.csv'
            self.probabilities[['PositionalProbability', 'Disposition']].to_csv(outfile)
            # Save the sources to be reused
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
        
        self.probabilities_generated = True

    def generate_priors(self, detection_file, periodrange=[0.5, 16], radiusrange=[1, 16], infile=None, rerun=False):
        """
        radiusrange only applies to planet scenarios. Should be the range simulated in training set creation (NOT the range of real candidates)
        periodrange should be the range used to run the BLS, both on the training sets to get detectionfactors and on the real data.
        periodrange can actually be longer than is sensible for a TESS sector, for example, as this will just reduce the detectionfactors. 
        However, periodrange cannot be larger than the range simulated for the training sets.
        EB scenarios assume all realistic stars are detectable, and limit themselves to main sequence stars.
        """
        if not self.sectordata:
            raise ValueError('Run generate_per_sector_data first')
        
        if not self.probabilities_generated:
            raise ValueError('Run generate_probabilities first')
        
        if rerun:
            self.priors = pd.DataFrame()
        elif infile is not None:
            self.priors = pd.read_csv(Path(infile)).set_index(['ticid', 'candidate'])
        elif self.load_suffix is not None:
            try:
                self.priors = pd.read_csv(self.output/f'Priors_{self.load_suffix}.csv').set_index(['ticid', 'candidate'])
            except FileNotFoundError:
                print(f'Priors file not found! Recreating...')
                self.priors = pd.DataFrame()

        if len(self.priors) > 0:
            tics_to_run = self.data.index.difference(self.priors.index)
            tics_to_run = tics_to_run.unique('ticid')
        else:
            tics_to_run = self.data.index.unique('ticid')
            
        num_targets = len(tics_to_run)
        
        if num_targets == 0:
            return
        
        from astropy.io import fits
        
        detectionfactors = pd.read_csv(f'TrainingSet/Output/{detection_file}', usecols=['sim', 'adj_detection']).set_index('sim')

        self.universal_priors = priorutils.calculate_universal_priors(detectionfactors, periodrange, radiusrange)

        self.densitymaps = {}
        for mapkey in [21.75, 21.5, 21.25, 21.0, 20.75, 20.5, 20.25, 20.0, 19.75, 19.5, 19.25, 19.0, 18.75, 18.5, 18.25,
                       18.0, 17.75, 17.5, 17.25, 17.0, 16.75, 16.5, 16.25, 16.0, 15.75, 15.5, 15.25, 15.0, 14.75, 14.5,
                       14.25, 14.0, 13.75, 13.5, 13.25, 13.0, 12.75, 12.5, 12.25, 12.0, 11.75, 11.5, 11.25, 11.0, 10.75,
                       10.5, 10.25, 10.0, 9.75, 9.5, 9.25, 9.0]:
            density_map = fits.open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Priors',
                                                           'trilegal_density_' + str(mapkey) + '_nside16.fits'))
            self.densitymaps[mapkey] = density_map[1].data['T'].flatten()

            density_map.close()
            
            
        priors_lst = []
        print(f'Generating priors for {len(self.data.loc[tics_to_run])} candidates')
        if self.multiprocessing > 1 and num_targets > 5:
            if num_targets < self.multiprocessing:
                workers = num_targets
            else:
                workers = self.multiprocessing
            
            factor = 15
            while num_targets < factor*workers:
                factor -= 1
                if factor == 1:
                    break
            
            ticid_split = np.array_split(tics_to_run, factor*workers)
            
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self.multi_priors, ticid_group):ticid_group for ticid_group in ticid_split}
                for future in as_completed(futures):
                    try:
                        priors_df, failed = future.result()
                        priors_lst.append(priors_df)
                        for fail in failed:
                            print(f'Exception {fail[1]} occured for ticid {fail[0]}.')
                    except Exception as e:
                        group = futures[future]
                        print(f'Exception {e} for ticid group: {group}.')       
        else:
            for targetid in tics_to_run:
                try:
                    priors_lst.append(self.candidate_priors(targetid))
                except Exception as e:
                    print(f'Exception: {e} occur for {targetid}')
       
        new_priors = pd.DataFrame(pd.concat(priors_lst))
       
        if len(self.priors) > 0:
            self.priors = pd.concat([self.priors, new_priors])
        else:
            self.priors = new_priors
        
        self.priors.sort_index(inplace=True)             
        
        if self.save_output:
            self.priors.to_csv(self.output / f'Priors_{self.save_suffix}.csv')
            
    def multi_priors(self, target_ids):
        priors_lst = []
        failed = []
        for ticid in target_ids:
            try:
                priors_df = self.candidate_priors(ticid)
                priors_lst.append(priors_df)
            except Exception as e:
                failed.append((ticid, e))
        
        try:    
            priors_df = pd.concat(priors_lst)
        except ValueError:
            priors_df = pd.DataFrame()
        
        return priors_df, failed
               
    def candidate_priors(self, ticid):
        gaia_rad = 2.2
        source_obj = self.sources[ticid]
            
        ra = source_obj.coords[0]  # for trilegal / beb density
        dec = source_obj.coords[1]
        
        priors_lst = []
        for candidate in self.data.loc[ticid].index:
            cndt_depth = self.sector_data.loc[(ticid, candidate), 'sec_depth'].mean()
            # takes magnitude of all flux in aperture, works out limiting magnitude of single system necessary to cause the observed eclipse
            mag = source_obj.Tmag
            maglim_beb = mag - 2.5 * np.log10(cndt_depth*1e-6)  # in relative flux for eclipse depth. *2 as max binary eclipse is assumed to be 50%
            maglim_btp = mag - 2.5 * np.log10(10 * cndt_depth*1e-6)  # in relative flux for eclipse depth. *10 as max planet eclipse is assumed to be 2%. NB will bug if real transits larger than 2% considered.

            if maglim_beb < mag:
                maglim_beb = None
            
            if maglim_btp < mag:
                maglim_btp = None

            cndt_priors = priorutils.calculate_candidate_priors(self.probabilities.loc[(ticid, candidate, ticid), 'PositionalProbability'], self.universal_priors,
                                                                maglim_beb, maglim_btp, gaia_rad, ra, dec,
                                                                self.densitymaps)
            
            # Check if no Nearby Source was identified a possible Nearby HEB
            if len(source_obj.nearby_assessment.query('Possible_NHEB == True')) == 0:
                # If so set, NHEB prior to 0
                cndt_priors['NHEB'] = 0
            # Check if no Nearby Source was identified a possible Nearby Transiting Planet
            if len(source_obj.nearby_assessment.query('Possible_NTP == True')) == 0:
                # If so set, NTP prior to 0
                cndt_priors['NTP'] = 0
                
            # Check the Gaia RUWE score and adjust the priors for the hierarchical secnarios on the target (HTP & HEB)
            ruwe = source_obj.ruwe
            
            if np.isnan(ruwe):
                pass
            elif ruwe <= 1.05:
                # Reduce the prior
                cndt_priors['HEB'] = (cndt_priors['HEB']/0.4)*0.28
                cndt_priors['HTP'] = (cndt_priors['HTP']/0.3)*0.21
            elif ruwe >= 1.4:
                # Boost the prior
                cndt_priors['HEB'] = (cndt_priors['HEB']/0.4)
                cndt_priors['HTP'] = (cndt_priors['HTP']/0.3)
                
            # Add the ticid and candidate to the dictionary to serve as the index later on
            cndt_priors['ticid'] = ticid
            cndt_priors['candidate'] = candidate
            
            priors_lst.append(cndt_priors)
        
        # Construct a dataframe from the list of dictionaries
        priors = pd.DataFrame(priors_lst).set_index(['ticid', 'candidate'])
        
        return priors


    def generate_features(self, infile_features=None, infile_som=None, rerun=False, mask_multi=False):
        """
        Calculate the LC related features for this candidate set
        """
        # Also check if nearby stars have been identified. Raise error if not.
        if not self.find_stars:
            raise ValueError('Run generate_sources first!')
        
        self.mask_multi = mask_multi
        
        # Read already computed features from previous run, if relevant input file specified
        if rerun:
            self.features = pd.DataFrame()
            self.som_array = np.array([])
        elif infile_features is not None:
            if infile_som is None:
                raise ValueError('Provide the corresponding som_array for the input feature file!')

            self.features = pd.read_csv(Path(infile_features)).set_index(['ticid', 'candidate'])
            self.som_array = np.load(Path(infile_som))
        elif self.load_suffix is not None:
            try:
                self.features = pd.read_csv(self.output/f'Features_{self.load_suffix}.csv').set_index(['ticid', 'candidate'])
                self.som_array = np.load(self.output/f'SOM_array_{self.load_suffix}.npy')
            except FileNotFoundError:
                print(f'Features or som_array file with provided suffix: {self.load_suffix} not found! Recreating...')
                self.features = pd.DataFrame()
                self.som_array = np.array([])
            except KeyError:
                print(f'Features with suffix: {self.load_suffix} do not have the required ticid and candidate data or are empty! Recreating...')
                self.features = pd.DataFrame()
                self.som_array = np.array([])
 
        if len(self.features) > 0:
            candidates_to_run = self.data.index.difference(self.features.index)
        else:
            candidates_to_run = self.data.index
        
        if len(candidates_to_run) == 0:
            return
        
        candidates_to_run = self.data.loc[candidates_to_run.unique('ticid')].index 

        df_lst = []
        som_array_lst = []
        # Output message for commencing feature generation.
        print(f'Generating features for {len(candidates_to_run)} candidates')  
        num_targets = len(candidates_to_run.unique('ticid'))
        if self.multiprocessing > 1 and num_targets > 5:
            if self.multiprocessing > num_targets:
                workers = num_targets
            else:
                workers = self.multiprocessing
            
            factor = 10
            while num_targets/(factor*workers) < 8:
                factor -= 1
                if factor == 1:
                    break
                
            ticid_split = np.array_split(candidates_to_run.unique('ticid'), factor*workers)
            
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self.multi_features, ticid_group):ticid_group for ticid_group in ticid_split}
                
                for future in as_completed(futures):
                    try:
                        features_df, som_arrays, failed = future.result()
                        df_lst.append(features_df)
                        som_array_lst.extend(som_arrays)
                        for fail in failed:
                            print(f'Exception {fail[1]} occured for ticid {fail[0]}.')
                    except Exception as e:
                        group = futures[future]
                        print(f'Exception {e} for ticid group: {group}.') 
        else:
            for target_id in candidates_to_run.unique('ticid'):
                try:
                    features_df, som_arrays = self.single_target_features(target_id)
                    df_lst.append(features_df)
                    som_array_lst.extend(som_arrays)
                except Exception as e:
                    print(f'Exception {e} for ticid {target_id}.')
                    
        if len(df_lst) > 0:           
            new_df = pd.concat(df_lst)
            new_som_array = np.array(som_array_lst)
            
            if len(self.features) > 0:
                self.features = pd.concat([self.features, new_df])
                self.som_array = np.concatenate([self.som_array, new_som_array])
            else:
                self.features = new_df
                self.som_array = new_som_array

        if len(self.features) > 0:
            self.features['sort_idx'] = np.arange(len(self.features), dtype=int)
            self.features.sort_index(inplace=True)
            self.som_array = self.som_array[self.features['sort_idx'].values]
            self.features.drop('sort_idx', axis=1, inplace=True)
            
            if self.save_output:
                self.features.to_csv(self.output / f'Features_{self.save_suffix}.csv')
                np.save(self.output / f'SOM_array_{self.save_suffix}.npy', self.som_array)
        else:
            print('No features were generated!')
        

    def multi_features(self, target_ids):
        feat_lst = []
        failed = []
        som_array_lst = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for ticid in target_ids:
                try:
                    features_df, som_arrays = self.single_target_features(ticid)
                    feat_lst.append(features_df)
                    som_array_lst.extend(som_arrays)
                except Exception as e:
                    failed.append((ticid, e))
        
        try:    
            features_df = pd.concat(feat_lst)
        except ValueError:
            features_df = pd.DataFrame()
        
        return features_df, som_array_lst, failed
        
        
    def single_target_features(self,target_id):
        tce_feat_lst = []
        som_array_lst = []
        
        source = self.sources[target_id]

        rstar = source.radius
        tmag = source.Tmag
        gmag = source.Gmag
        teff = source.teff
        dist =source.dist
        bp_rp = source.bp_rp

        target_data = self.data.loc[target_id]
        
        target_sector_data = self.sector_data.loc[target_id].copy()
        target_sector_data.drop(['t0', 'per', 'sec_time','sec_flux', 'sec_f'], axis=1, inplace=True)
        
        sec_data = target_sector_data.drop('sec_C', axis=1).groupby('candidate').median()
        sec_data['sec_C'] = target_sector_data[['sec_C']].groupby('candidate').mean()
        
        sec_data.rename(columns={'new_t0':'t0',
                            'new_per':'per',
                            'sec_tdur':'tdur',
                            'sec_tdur23':'tdur23',
                            'sec_depth':'depth',
                            'sec_C':'target_fraction'}, inplace=True)
        
        tce_pers = sec_data.per.values
        tce_t0s = sec_data.t0.values
        tce_tdurs = sec_data.tdur.values
    
        sectors = target_data.iloc[0].sectors

        fluxes = np.array([])
        times = np.array([])
        errors = np.array([])
        
        lcs = {}
        for sec in sectors:
            lc_file = cutils.lc_filepath(self.lc_dir, self.dir_style, target_id, sec)
            try:
                lc = cutils.load_spoc_lc(lc_file, flatten=True, transitcut=True,
                                         tc_per=tce_pers, tc_t0=tce_t0s, tc_tdur=tce_tdurs)
            except FileNotFoundError:
                continue
            
            lcs[sec] = lc
            
            fluxes = np.concatenate([fluxes, lc['flux']])
            times = np.concatenate([times, lc['time']])
            errors = np.concatenate([errors, lc['error']])

        lc = {'flux':fluxes,
              'time':times,
              'error':errors}
        
        for tce in target_data.index:

            candidate_feats = {}
            
            flag = ''
            
            per = sec_data.loc[tce, 'per']
            t0 = sec_data.loc[tce, 't0']
            tdur = sec_data.loc[tce, 'tdur']
            if per == 0:
                per = lc['time'][-1] - (t0-tdur*0.5)
                flag += '0_per '
            depth = sec_data.loc[tce, 'depth']
            tdur_p = tdur/per
            
            # if source.Gaia_rad_flag:
            #     flag += 'gaia_radius '
            # if source.Gaia_teff_flag:
            #     flag += 'gaia_teff'
            if source.no_gaia_id:
                flag += 'no_dr2_id '
            if source.Gaia_dr3_flag:
                flag += source.Gaia_dr3_flag 

            candidate_feats['ticid'] = target_id
            candidate_feats['candidate'] = tce
            candidate_feats['flag'] = flag
            candidate_feats['Tmag'] = tmag
            candidate_feats['Gmag'] = gmag
            candidate_feats['BP-RP'] = bp_rp
            candidate_feats['dist'] = dist
            candidate_feats['rstar'] = rstar
            candidate_feats['teff'] = teff
            candidate_feats['per'] = per
            candidate_feats['t0'] = t0
            candidate_feats['tdur'] = tdur
            candidate_feats['tdur_per'] = tdur_p
            candidate_feats['depth'] = depth

            try:
                disp = target_data.loc[tce, 'tfop_disp']
                if pd.isna(disp):
                    disp = target_data.loc[tce, 'tess_disp']
            except KeyError:
                disp = 'TCE'
                
            candidate_feats['disp'] = disp

            tce_lc = copy(lc)
            
            if self.mask_multi and len(target_data.index) > 1:
                mask = target_data.index.unique('candidate') != tce
                other_pers = tce_pers[mask]
                other_t0s = tce_t0s[mask]
                other_tdurs = tce_tdurs[mask]
                        
                for i in range(len(other_pers)):
                    tce_lc = futils.transit_cut(tce_lc, other_pers[i], other_t0s[i], other_tdurs[i])
                
            if lc:
                obs_transits = futils.observed_transits(lc['time'], t0, per, tdur)
                if obs_transits > 0:
                    if futils.check_transit_points(lc['time'], per, t0, tdur):
                        fill_nan = False
                    else:
                        print(f'No transit points found for {target_id}-{tce}')
                        fill_nan = True
                else:
                    print(f'Candidate {target_id}-{tce} has no transits in the observation window.')
                    fill_nan = True

                candidate_feats['transits'] = obs_transits
                candidate_feats = fc.generate_LCfeatures(tce_lc, candidate_feats, per, t0, tdur,
                                                        depth, rstar, teff, fill_nan=fill_nan)
            else:
                fill_nan = True
                candidate_feats['transits'] = np.nan
                candidate_feats = fc.generate_LCfeatures(tce_lc, candidate_feats, per, t0, tdur, depth, rstar, teff, fill_nan=fill_nan)

            if fill_nan:
                candidate_feats['target_fraction'] = np.nan
                candidate_feats['nearby_fraction'] = np.nan
            else:
                candidate_feats['target_fraction'] = sec_data.loc[tce, 'target_fraction']
                candidate_feats['nearby_fraction'] = source.nearby_fractions.drop(target_id).mean(axis=1).max()
                
            tce_feat_lst.append(candidate_feats)

            som_array = TSOM.PrepareOneLightcurve(None, per, t0, candidate_feats['fit_tdur'], nbins=48, clip_outliers=5, lc=np.array([tce_lc['time'], tce_lc['flux'], tce_lc['error']]).T)
            som_array_lst.append(som_array)
        
        target_features = pd.DataFrame(tce_feat_lst).set_index(['ticid', 'candidate'])
        return target_features, som_array_lst
    
    
    def classify_candidates(self,clfs, model_loc='default', model_suffix='', train_suffix1='training', train_suffix2='training',
                            som_shape=(20,20), som_array_length=48, transform=True, uniform=False):
        
        classet = ML_Classification.Classify(self.features, self.som_array, clfs, self.output, self.output, model_loc, model_suffix, 
                                             train_suffix1, train_suffix2, som_shape, som_array_length, transform, uniform,
                                             self.save_output, self.save_suffix)
        
        classet.multi_scenario_classification()  

        self.candidate_classifications = classet.classifications
       
        if self.save_output:
            if uniform:
                self.candidate_classifications.to_csv(self.output / f'ML_Classifications_mean_uniform_{self.save_suffix}.csv')
            else:
                self.candidate_classifications.to_csv(self.output / f'ML_Classifications_mean_{self.save_suffix}.csv')

        
    def validation(self, clf='mean', uniform=False):
        # Results list
        final_probabilities = []
        
        # Retrieve the scenarios for which ML probabilities were produced
        scenarios = self.candidate_classifications.columns
        # Compute posterior probabilities for each candidate
        for ticid, candidate in self.candidate_classifications.index:
            try:
                # Define results container for ease of saving
                posteriors = {}
                posteriors['ticid'] = ticid
                posteriors['candidate'] = candidate

                # Loop through the fp scenarios
                for fp in scenarios:
                    # Retrieve the ML classification probability for Planet-vs-FP
                    ml_prob =  self.candidate_classifications.loc[(ticid, candidate), fp]
            
                    if fp == 'NSFP':
                        # The NSFP scenario has no prior probability.
                        # Hence the posterior probability is equal to the ML probability.
                        scenario_posterior = ml_prob
                    else:
                        # Use Bayesian inference to compute the posterior probability by updating the ML derived probability 
                        # with the candidate prior probability for the planet and the FP scenario
                        fp_prior = self.priors.loc[(ticid, candidate), fp]
                        pla_prior = self.priors.loc[(ticid, candidate), 'Planet']
                        if fp_prior == 0:
                            # If the fp prior is 0, then the candidate planet posterior against this scenario is 1.0
                            posteriors[fp] = 1
                            continue
                        # Compute the prior informed planet and fp probabilities
                        pla_prob = ml_prob * pla_prior
                        fp_prob = (1 - ml_prob) * fp_prior
                        # Bayesian inference to compute the Planet posterior probability against the specific FP scneario
                        scenario_posterior = pla_prob/(pla_prob+fp_prob)
                    
                    # Store the planet-vs-FP scenario posterior probability
                    posteriors[fp] = scenario_posterior
                
                # Store all candidate probability                 
                final_probabilities.append(posteriors)
            except Exception:
                continue
        
        # Transform the results into a dataframe for ease of use and saving    
        self.validation_results = pd.DataFrame(final_probabilities).set_index(['ticid', 'candidate'])
        
        # Save the results if requested
        if self.save_output:
            if uniform:
                self.validation_results.to_csv(self.output / f'Validation_{clf}_uniform_{self.save_suffix}.csv')
            else:
                self.validation_results.to_csv(self.output / f'Validation_{clf}_{self.save_suffix}.csv')
        
class Source(object):

    def __init__(self, tic, coords=(), pm=(), dist=None, Tmag=None, Gmag=None, radius=None, mass=None, teff=None):
        """
        Datastore for a target source
        """
        self.TIC = tic
        self.coords = coords
        self.pm = pm
        self.dist = dist
        self.Tmag = Tmag
        self.Gmag = Gmag
        self.radius = radius
        self.mass = mass
        self.teff = teff
        self.ruwe = np.nan
        self.bp_rp = np.nan
        self.no_gaia_id = 0
        self.Gaia_dr3_flag = 0
        # self.Gaia_rad_flag = 0
        # self.Gaia_teff_flag = 0
        self.non_gaia_sources = 0
        self.scc = pd.DataFrame(columns=['sector', 'cam', 'ccd']).set_index('sector')
        self.wcs = {}
        self.origin = {}
        self.aperture_mask = {}
        self.centroid_mask = {}
        self.cor_factor = {}
        self.nearby_data = None
        self.nearby_fractions = None
        self.nearby_depths = pd.DataFrame()
        self.totalflux = {}
        self.totalmag_equivalent = {}
        self.prfs = {}
        self.cent_out = pd.DataFrame(columns=['sector', 'X', 'Y']).set_index('sector')
        self.cent_in = pd.DataFrame()
        self.model_centroid = pd.DataFrame()
        self.prob_centroid = pd.DataFrame()
        self.nearby_assessment = None
        self.trilegal_density_beb = None
        self.trilegal_density_btp = None
