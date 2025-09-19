from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from Features.TransitSOM import TransitSOM
from TrainingSet.RunSOM import load_training_som_array
from TrainingSet.utils import load_training_ids, load_trained_GP, load_trained_XGB, load_pickled_model
from Features import utils as futils

class Classify(object):

    def __init__(self, features_input, som_array_input, clfs, input_loc='default', output_loc='default', model_loc='default', model_suffix='', 
                 train_suffix1='training', train_suffix2='training', som_shape=(20,20), som_array_length=48, transform=True, uniform=False, save_output=False, save_suffix=''):
    
        # Define paths for input, output and model locations
        if input_loc == 'default':
            self.input_loc = Path(__file__).resolve().parents[1] / 'Output' 
        else:
            self.input_loc = Path(input_loc) 
                    
        if output_loc == 'default':
            self.output = Path(__file__).resolve().parents[1] / 'Output' 
        else:
            self.output = Path(output_loc)
            
        if model_loc == 'default':
            self.model_loc = Path(__file__).resolve().parents[1] / 'TrainingSet' / 'Output'
        else:
            self.model_loc = Path(model_loc) 
            
        self.model_suffix = model_suffix
        
        # Define the classifiers to be used
        if not isinstance(clfs, list):
            self.clfs = [clfs]
        else:
            self.clfs = clfs  
        
        # Load candidate features
        if type(features_input) == pd.DataFrame:
            self.features = features_input
        else:
            self.features = pd.read_csv(self.input_loc / features_input).set_index(['ticid', 'candidate'])
        
        # Load candidate SOM arrays
        if type(som_array_input) == np.ndarray:
            self.som_array = som_array_input
        else:
            self.som_array = np.load(self.input_loc / som_array_input)
                    
        # Define the map from simulation name to scenario name
        self.sim_to_scenario = {'PLA':'Planet',
                                'uPLA':'Planet',
                                'EB':'EB',
                                'BEB':'BEB',
                                'NEB':'NEB',
                                'TRIPLE':'HEB',
                                'NTRIPLE':'NHEB',
                                'NPLA':'NTP',
                                'NuPLA':'NTP',
                                'PIB':'HTP',
                                'uPIB':'HTP',
                                'BTP':'BTP',
                                'uBTP':'BTP',
                                'NSFP':'NSFP'}
        
        # Set the training set related parameters
        self.train_suffix1 = train_suffix1
        self.train_suffix2 = train_suffix2
        self.som_shape = som_shape
        self.som_array_length = som_array_length
        self.transform = transform
        self.uniform = uniform
        
        # Set output parameters
        self.save_output = save_output
        self.save_suffix = save_suffix
        
        # Define output dataframes
        self.classifier_probabiltiies = {clf:pd.DataFrame(index=self.features.index) for clf in self.clfs}
            
        self.classifications = pd.DataFrame(index=self.features.index)
        self.outliers = pd.DataFrame(index=self.features.index)
        
        
    def run_scenario_classification(self, sim1, sim2):
        
        features = self.features.copy()
        
        features['TmagGmag'] = features['Tmag'] - features['Gmag']
        
        # Compute SOM features
        try:
            som_file = self.model_loc / f'{sim1}-{sim2}_SOM{self.model_suffix}.txt'
            som = TransitSOM.LoadSOM(som_file, self.som_shape[0], self.som_shape[1], self.som_array_length)
        except FileNotFoundError:
            raise RuntimeError(f'Trained SOM: "{som_file}" not found! Generate SOM from training sets first.')
            
        try:
            som_proportions = np.load(self.model_loc / f'{sim1}-{sim2}_SOM_proportions{self.model_suffix}.npy')
        except FileNotFoundError:
            print('Pre-computed SOM proportions not found, generating from training sets...')
            train_1, train_2 = load_training_ids(sim1, sim2, self.train_suffix1, self.train_suffix2, load_suffix=self.model_suffix)
            
            training_array = load_training_som_array(sim1, sim2, train_1, train_2, load_suffix=self.model_suffix)
            
            training_groups = np.concatenate([np.zeros(len(train_1), dtype=int), np.ones(len(train_2), dtype=int)])

            som_proportions = TransitSOM.generate_proportions(som, training_array, training_groups, map_shape=self.som_shape, 
                                                              outfile= self.model_loc / f'{sim1}-{sim2}_SOM_proportions{self.model_suffix}.npy')
        
        som_prob, som_dist = TransitSOM.classify(som, som_proportions, self.som_array)
        
        features['SOM'] = som_prob
        features['SOM_dist'] = som_dist
        
        if not (sim1 in ['NPLA','NuPLA', 'NEB','NTRIPLE'] or sim2 in ['NPLA','NuPLA', 'NEB','NTRIPLE']):
            features.drop(['target_fraction', 'nearby_fraction'], axis=1, inplace=True)
                
        # Prepare the features and separate them into training and test sets
        features = futils.prepare_features(features)

        if self.transform:
            # Load quantile transformer
            infile = self.model_loc / f'{sim1}-{sim2}{self.model_suffix}_transformer.pkl'
            qt = load_pickled_model(infile)
                
            # Apply the transformation
            features[qt.feature_names_in_] = qt.transform(features[qt.feature_names_in_])
        
        if sim1 == 'NSFP' or sim2 == 'NSFP':
            drop_cols = ['fit_chisq', 'rsnrmes', 'min_mes', 'mad_mes', 'median_mes']
        else:
            drop_cols = ['fit_chisq', 'rminmes', 'rmesmed', 'rmesmad', 'rsnrmes', 'min_mes', 'mad_mes', 'median_mes']
        
        features.drop(drop_cols, axis=1, inplace=True)
                    
        scenario_probabilities = []

        for clf in self.clfs:  
        # Load the classifier
            if clf == 'rfc':
                # Set the model file name
                model_file = f'{sim1}-{sim2}{self.model_suffix}_{clf}_calibrated.pkl'
                # Load the model
                model = load_pickled_model(model_file)
                
                # Compute the ML posterior probabilities for the scenario
                clf_scenario_probs = model.predict_proba(features)[:,1]
            elif clf == 'GP':
                # Set the model file name
                model_file = f'{sim1}-{sim2}{self.model_suffix}_{clf}.pt'
                # Load the model
                model = load_trained_GP(self.model_loc / model_file)
                # Transform the features dataframe to an array as required by the GP model
                clf_features = features.to_numpy().astype(np.float32)
                # Compute the ML posterior probabilities for the scenario
                clf_scenario_probs = model.predict_proba(clf_features)[:,1]
            elif clf == 'xgb':
                # Set the model file name
                model_file = f'{sim1}-{sim2}{self.model_suffix}_{clf}.json'
                # Load the model
                model = load_trained_XGB(self.model_loc / model_file)
                # Compute the ML posterior probabilities for the scenario
                clf_scenario_probs = model.predict_proba(clf_features)[:,1]
            
            # Store the scenario probabilities for the classifier
            self.classifier_probabiltiies[clf][self.sim_to_scenario[sim2]] = clf_scenario_probs
                
            scenario_probabilities.append(clf_scenario_probs)
            
            if self.save_output:
                if self.uniform:
                    self.classifier_probabiltiies[clf].to_csv(self.output / f'ML_Classifications_{clf}_uniform_{self.save_suffix}.csv')
                else:
                    self.classifier_probabiltiies[clf].to_csv(self.output / f'ML_Classifications_{clf}_{self.save_suffix}.csv')
        
        # Get the mean scenario probabilities across the classifiers
        self.classifications[self.sim_to_scenario[sim2]] = np.mean(np.array(scenario_probabilities), axis=0)
        
        
    def multi_scenario_classification(self, primary_sim='PLA'):
        if self.uniform:
            sims = ['uPLA', 'EB', 'TRIPLE', 'BEB', 'uPIB', 'uBTP', 'NuPLA', 'NEB', 'NTRIPLE', 'NSFP']
        else:
            sims = ['PLA', 'EB', 'TRIPLE', 'BEB', 'PIB', 'NPLA', 'NEB', 'NTRIPLE', 'NSFP']
        
        if self.uniform and primary_sim == 'PLA':
            primary_sim = 'uPLA'
            
        sims.remove(primary_sim)
        
        for secondary_sim in sims:
            self.run_scenario_classification(primary_sim, secondary_sim)
                
                
    def outlier_detection(self, sim1, sim2, train_suffix1='training', train_suffix2='training'):
        features = self.features.copy()
        # Compute SOM features
        train_1, train_2, training_array = load_training_som_array(sim1, sim2, train_suffix1, train_suffix2, load_suffix=self.model_suffix)
        
        training_groups = np.concatenate([np.zeros(len(train_1), dtype=int), np.ones(len(train_2), dtype=int)])
        som = TransitSOM.LoadSOM(self.model_loc / f'{sim1}-{sim2}_SOM{self.model_suffix}.txt', 20, 20, 48) 
        
        som_prob, som_dist = TransitSOM.classify(som, training_array, training_groups, self.som_array)
        
        features['SOM'] = som_prob
        features['SOM_dist'] = som_dist
        
        if not (sim1 in ['NPLA','NEB','NTRIPLE'] or sim2 in ['NPLA','NEB','NTRIPLE']):
            features.drop(['target_fraction'], axis=1, inplace=True)
                
        # Prepare the features and separate them into training and test sets
        features = futils.prepare_features(features)
                
        infile = self.model_loc / f'{sim1}-{sim2}{self.model_suffix}_outliers.pkl'
        with open(infile, 'rb') as f:
            model = pickle.load(f)
            
        decision = model.decision_function(features)
        
        self.outliers[self.sim_to_scenario[sim2]] = decision
        
        if self.save_output:
            self.outliers.to_csv(self.output / f'ML_Outliers_{self.save_suffix}.csv')


    def multi_scenario_outliers(self, primary_sim='PLA'):
        sims = ['PLA', 'EB', 'TRIPLE', 'BEB', 'PIB', 'BTP', 'NPLA', 'NEB', 'NTRIPLE', 'NSFP']
        sims.remove(primary_sim)
        
        for secondary_sim in sims:
            if secondary_sim in ['PIB', 'BTP', 'NPLA', 'NSFP']:
                self.outlier_detection(primary_sim, secondary_sim, train_suffix1=secondary_sim)
            else:
                self.outlier_detection(primary_sim, secondary_sim)