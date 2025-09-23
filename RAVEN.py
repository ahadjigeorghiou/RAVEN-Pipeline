'''
Script that runs the RAVEN pipeline on a single target or a collection of targets with their associated candidates.

To run the pipeline, place the lightcurve files associated with the target star in the Lightcurve/ folder.
Then add a csv file with the candidate transit data in the Input/ folder. 

The candidates must:
1) have at least one associated SPOC FFI lightcurve up to Sector 55
2) have a period between 0.5 and 16 days.
3) A depth greater than 300ppm.

Call run_pipeline('{csv filename}') to apply the pipeline on your candidates within a python script or in the command line with:
python RAVEN.py {csv filename}

Additional input options are available. See the documentation of the run_pipeline function for more details.
'''


from CandidateSet import CandidateSet as cs
import argparse as ap


def run_pipeline(infile, infile_type='default', lc_directory='default', dir_structure='single', save_suffix=None, load_suffix=None, output_directory='default', num_process=0):
    """
    Runs the full RAVEN pipeline for a given set of candidates.
    
    The input file should be a csv or txt file placed in the Input folder with the following columns:
    [ticid, candidate, per, t0, tdur, depth]
    
    The depth must be greater than 300ppm and the period must lie within the period range of 0.5d and 16d.
    
    An absolute path to an input file can also be provided.
    
    The input file can also be:
    1) a dataframe object with the above columns present. Infile type must be set to 'dataframe'.
    2) a TOI list from the Exoplanet Archive. Infile type must be set to 'archive'.
    3) a TOI list from ExoFOP. Infile type must be set to 'ExoFOP'.
    
    The default lightcurve directory

    Parameters
    ----------
    infile : str or pandas.DataFrame
        - Name of the input file placed in the Input/ directory
        - Absolute path to the input file
        - pandas.DataFrame object containing the candidate data
    infile_type : {'default', 'archive', 'exofop', 'recovery', 'dataframe'}, optional
        Specifies the format of the input data:
        - 'default': User-generated CSV with columns: [ticid, candidate, per, t0, tdur, depth]
        - 'archive': TOI list as downloaded from NASA Exoplanet Archive
        - 'exofop': TOI list as downloaded from ExoFOP
        - 'recovery': Candidate list from RAVEN's built-in BLS survey
        - 'dataframe': Input is a pandas DataFrame object
    lc_directory : str, optional
        Path to the directory containing the lightcurve files. Defaults to the 'Lightcurves/'
        folder in the project directory. 
    dir_structure : {'per_target', 'spoc', 'single'}, optional
        Organization structure of the lightcurve directory:
        - 'single': All lightcurve files in one directory
        - 'per_target': Organized with one sub-folder per TIC ID containing all its lightcurves
        - 'spoc': Lightcurves organized exactly as downloaded from SPOC sector releases
    save_suffix : str, optional
        A suffix to append to all saved output files for this run. If None, the input filename
        will be used unless the input is a dataframe.
    load_suffix : str, optional
        A suffix to identify and load files from a previous run, allowing the
        pipeline to resume or reuse prior results. Defaults to None.
    output_directory : str or Path, optional
        Path to a user defined output directory. Default is the 'Output/' folder in the project directory.
    num_process : int, optional
        The number of processes to use for multiprocessing tasks. If 0, multiprocessing is disabled. 
        Defaults to 0.

    Returns
    -------
    CandidateSet
        An instance of the CandidateSet class containing all processed data and results from the pipeline run.
    """
    
    
    # Initiate the pipeline class instance
    toi_cset = cs.CandidateSet(infile, 
                               infile_type, 
                               lc_dir=lc_directory, 
                               dir_structure=dir_structure, 
                               save_output=True, 
                               save_suffix=save_suffix, 
                               load_suffix=load_suffix,
                               output_directory=output_directory,
                               multiprocessing=num_process, 
                               per_lim=[0.5, 16], depth_lim=300)

    # Check what lightcurves have been made available by the user
    toi_cset.check_target_lcs(donwload_missing=False)
    
    
    # Generate the source objects for each TIC star in the input
    toi_cset.generate_sources()
    
    print(f'Source generation completed!')
    
    # Remove from the input data sources for which TIC and Gaia data were not retrieved  
    toi_cset.data = toi_cset.data.loc[toi_cset.sources.keys()]

    # Compute the per sector data for each candidate
    toi_cset.generate_per_sector_data(skip_per_correction=True)
    
    print(f'Sector Data generated!')
    
    # Determine the observed centroid offset of each candidate in each sector of observation
    toi_cset.generate_centroiddata()
    
    print(f'Centroid Data generated!')

    # Estimate the flux fractions of the target and the known nearby sources in the SPOC aperture
    toi_cset.estimate_flux_fractions(rerun=False)
    
    # Estimate the depth of the event on the known nearby sources based on their flux fractions
    toi_cset.estimate_nearby_depths(rerun=False)
    
    # Compute the Positional Probabilities
    toi_cset.generate_positional_probabilities()
    
    print(f'Positional Probabilities Generate!')
    
    # Compute the ML features
    toi_cset.generate_features(mask_multi=True)
    
    print(f'ML Features generated!')
    
    # Compute the scenario specific prior probabilities
    toi_cset.generate_priors(detection_file='detection_factors_New.csv')
    
    print(f'Prior Probabilities generated!')
    
    # Compute the ML posterior probabilities
    toi_cset.classify_candidates(['GP','XGB'], model_suffix='_New', transform=True, uniform=False)
    
    # Compute the final posterior probabilities
    toi_cset.validation(clf='mean', uniform=False)
    
    print('Done!')
    
    return toi_cset
    
    

def parse_cli_args():
    """
    Parses command-line arguments for running the RAVEN pipeline.
    """
    parser = ap.ArgumentParser(description='Run the RAVEN pipeline on a set of TESS candidates.')
    
    parser.add_argument('infile', type=str,
                        help='Name of the input file in the Input/ directory or a direct path to a file.')
    
    parser.add_argument('--infile_type', type=str, default='default',
                        choices=['default', 'archive', 'exofop', 'recovery', 'dataframe'],
                        help="Input data format: 'default' (user CSV with ticid,candidate,per,t0,tdur,depth), "
                             "'archive' (NASA Exoplanet Archive TOI list), 'exofop' (ExoFOP-TESS TOI list), "
                             "'recovery' (recovery test format), 'dataframe' (pandas DataFrame object).")
                        
    parser.add_argument('--lc_dir', type=str, default='default',
                        help="Path to the directory containing lightcurve files. Defaults to 'Lightcurves/'.")
                        
    parser.add_argument('--dir_structure', type=str, default='single',
                        choices=['single', 'per_target', 'spoc'],
                        help="Lightcurve directory organization structure: 'single' (all files in one directory), "
                             "'per_target' (one subfolder per TIC ID), "
                             "'spoc' (preserves SPOC release sector-based organization).")
                        
    parser.add_argument('--save_suffix', type=str, default=None,
                        help='Suffix to append to all saved output files. Defaults to the input filename.')
                        
    parser.add_argument('--load_suffix', type=str, default=None,
                        help='Suffix to identify and load files from a previous run.')
    
    parser.add_argument('--output_directory', type=str, default='default',
                        help="Path to a user defined output directory. Default is the 'Output/' folder in the project directory.")
                        
    parser.add_argument('--num_process', type=int, default=0,
                        help='Number of processes for multiprocessing. 0 disables it. Default is 0.')
                        
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli_args()
    
    result = run_pipeline(
        infile=args.infile,
        infile_type=args.infile_type,
        lc_directory=args.lc_dir,
        dir_structure=args.dir_structure,
        save_suffix=args.save_suffix,
        load_suffix=args.load_suffix,
        output_directory=args.output_directory,
        num_process=args.num_process
    )
    
    if len(result.flagged) > 0:
        print('The following candidates were flagged due to sector depth variations:')
        for ticid, candidate in result.flagged:
            print(f'{ticid}-{candidate}')
    
    print('Positional Probabilities:')    
    print(result.probabilities[['PositionalProbability', 'Disposition']])
    
    print('Posterior Probabilities:')   
    print(result.validation_results)
    
    