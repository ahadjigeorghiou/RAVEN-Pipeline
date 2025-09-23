RAVEN is an automated vetting and validation pipeline for TESS Exoplanet candidates that utilises ML models trained on comprehensive synthetic training sets. For further details on the pipeline implemenentation and performance testing please see:
https://arxiv.org/abs/2509.17645


Trained model files are provided so that users can apply the pipeline on their candidates. The current limitations are:
1) Period must be between 0.5 and 16 days
2) Depth must be greater than 300ppm
3) SPOC FFI lightcurves up to Sector 55

The pipeline is deployed online on https://huggingface.co/spaces/ahadjigeorghiou/RAVEN-Pipeline and can be used without any additional setup. 

To run the pipeline locally: 
1) Clone the project and install its requirements.
2) Create a csv file with the following data [ticid, candidate, per, t0, tdur, depth]
3) Place the csv file in {Project Root}/Input/
4) Place the associated lightcurves in {Project Root}/Lightcurves/
5) Run RAVEN.py {name_of_csv_file}

Alternatively, import and call the run_pipeline() function from RAVEN.py inside your python script. Additional run options are available.

