from pathlib import Path
import pandas as pd
import numpy as np
from Features.TransitSOM import TransitSOM
from TrainingSet.RunSOM import load_training_som_array
from TrainingSet import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from TrainingSet import GPC
from skorch.probabilistic import GPBinaryClassifier
from skorch.callbacks import EarlyStopping, PrintLog
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import torch
import xgboost as xgb
import json

class Training(object):

    def __init__(self, sim1, sim2, features_suffix='', train_suffix1='training', train_suffix2='training',
                 som_shape=(20,20), som_array_length=48, validation=False, 
                 input_loc='default', output_loc='default', model_suffix=None, 
                 match_testset=False, qtransform=True, quantiles=1000, load_transform=True, save_transform=True):

        self.sim1 = sim1
        self.sim2 = sim2
        
        self.features_suffix = features_suffix
                
        if input_loc == 'default':
            self.input_loc = Path(__file__).resolve().parents[1] / 'TrainingSet' / 'Simulations' 
        else:
            self.input_loc = Path(input_loc) 
                    
        if output_loc == 'default':
            self.output = Path(__file__).resolve().parents[1] / 'TrainingSet' / 'Output' 
        else:
            self.output = Path(output_loc)
            
        if model_suffix is None:
            self.model_suffix = self.features_suffix
        else:
            self.model_suffix = model_suffix

        # Load features
        features_1 = pd.read_csv(self.input_loc / f'{sim1}' / f'{sim1}_features{features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
        features_2 = pd.read_csv(self.input_loc / f'{sim2}' / f'{sim2}_features{features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
                
        features_1['TmagGmag'] = features_1['Tmag'] - features_1['Gmag']
        features_2['TmagGmag'] = features_2['Tmag'] - features_2['Gmag']
        
        # Load outliers files, if present, and remove them from the features
        try:
            outliers1 = pd.read_csv(self.input_loc / f'{sim1}' / f'{sim1}_outliers{features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
            
            features_1 = features_1.loc[features_1.index.difference(outliers1.index)]
        except FileNotFoundError:
            pass
        
        try:
            outliers2 = pd.read_csv(self.input_loc / f'{sim2}' / f'{sim2}_outliers{features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
            
            features_2 = features_2.loc[features_2.index.difference(outliers2.index)]
        except FileNotFoundError:
            pass
        
        # Load training ids
        train_1, train_2 = utils.load_training_ids(sim1, sim2, train_suffix1, train_suffix2, load_suffix=features_suffix, directory=self.input_loc)
                
        # Compute SOM features
        som = TransitSOM.LoadSOM(self.output / f'{sim1}-{sim2}_SOM{features_suffix}.txt', som_shape[0], som_shape[1], som_array_length)
        
        try:
            som_proportions = np.load(self.output / f'{sim1}-{sim2}_SOM_proportions{features_suffix}.npy')
        except FileNotFoundError:
            print('Pre-computed SOM proportions not found, generating from training sets...')

            training_array = load_training_som_array(sim1, sim2, train_1, train_2, load_suffix=features_suffix, directory=self.input_loc)
            
            training_groups = np.concatenate([np.zeros(len(train_1), dtype=int), np.ones(len(train_2), dtype=int)])

            som_proportions = TransitSOM.generate_proportions(som, training_array, training_groups, map_shape=som_shape, 
                                                              outfile=self.output / f'{sim1}-{sim2}_SOM_proportions{features_suffix}.npy')
        
        features_1 = self.som_feature(sim1, features_1, som, som_proportions)
        features_2 = self.som_feature(sim2, features_2, som, som_proportions)
        
        if not (sim1 in ['NPLA','NEB','NTRIPLE', 'NuPLA'] or sim2 in ['NPLA','NEB','NTRIPLE', 'NuPLA']):
            try:
                features_1.drop(['target_fraction', 'nearby_fraction'], axis=1, inplace=True)
            except KeyError:
                pass
            try:
                features_2.drop(['target_fraction', 'nearby_fraction'], axis=1, inplace=True)
            except KeyError:
                pass
               
        # Prepare the features and separate them into training and test sets
        features_1 = utils.prepare_training_sets(features_1, 1)
        features_2 = utils.prepare_training_sets(features_2, 0)
        
        # Separate the features for training
        trainset_1 = features_1.loc[train_1]
        features_1.drop(train_1, inplace=True)
        
        trainset_2 = features_2.loc[train_2]
        features_2.drop(train_2, inplace=True)
        
        testset_1 = features_1
        testset_2 = features_2
        
        if match_testset:
            if len(testset_1) > len(testset_2):
                testset_1 = testset_1.sample(len(testset_2), random_state=1)
            elif len(testset_2) > len(testset_1):
                testset_2 = testset_2.sample(len(testset_1), random_state=1)
                
                
        if validation:
            size = int(min((len(testset_1), len(testset_2)))*0.5)
            valset_1 = testset_1.sample(size, random_state=1)
            valset_2 = testset_2.sample(size, random_state=1)
            
            testset_1.drop(valset_1.index, inplace=True)
            testset_2.drop(valset_2.index, inplace=True)
        else:
            valset_1 = pd.DataFrame()
            valset_2 = pd.DataFrame()
                
        # Concatenate the arrays into one
        trainset = pd.concat([trainset_1, trainset_2])
        testset = pd.concat([testset_1, testset_2])
        valset = pd.concat([valset_1, valset_2])
        
        # Add sim_type to index
        trainset.set_index([trainset.index, 'sim'], inplace=True)
        testset.set_index([testset.index, 'sim'], inplace=True)
        try:
            valset.set_index([valset.index, 'sim'], inplace=True)
        except KeyError:
            pass
                
        # Shuffle them
        trainset = shuffle(trainset)
        testset = shuffle(testset)
        valset = shuffle(valset)
        
        # Separate the ground truth
        self.train_labels = trainset['disp'].astype(np.int8)
        trainset.drop('disp', axis=1, inplace=True)
        
        self.test_labels = testset['disp'].astype(np.int8)
        testset.drop('disp', axis=1, inplace=True)
        
        try:
            self.val_labels = valset['disp'].astype(np.int8)
            valset.drop('disp', axis=1, inplace=True)
        except KeyError:
            self.val_labels = None
            
        
        if qtransform:
            # Choose columns to be transformed
            cols = trainset.columns.drop(['grazestat', 
                                          'max_secmesp', 'SOM'])
            
            # Check if the transformer already exist:
            if load_transform:
                infile = self.output / f'{sim1}-{sim2}{self.model_suffix}_transformer.pkl'
                if infile.exists():
                    # Load quantile transformer
                    qt = utils.load_pickled_model(infile)
                    # Apply the transformer on the trainset
                    trainset[cols] = qt.transform(trainset[cols])
                    # As loaded an existing transformer, no need to save it again
                    save_transform = False
                    train_transform = False
                else:
                    train_transform = True
            else:
                train_transform = True
            
            if train_transform:
                # Otherwise set up the quantile transformer        
                qt = QuantileTransformer(n_quantiles=quantiles, output_distribution='normal')
                # Fit and transform the transformer based on the training set
                trainset[cols] = qt.fit_transform(trainset[cols])
            
            # Apply the transformation on the test and validation sets
            testset[cols] = qt.transform(testset[cols])
            try:
                valset[cols] = qt.transform(valset[cols])
            except KeyError:
                pass
                                
            # Save the transformer
            if save_transform:
                outfile = self.output / f'{sim1}-{sim2}{self.model_suffix}_transformer.pkl'
                utils.save_pickled_model(qt, outfile)
        else:
            qt = None
            
        # Remove some non important features
        if self.sim1 != 'NSFP' and self.sim2 != 'NSFP':
            cols = ['fit_chisq', 'rsnrmes', 'min_mes', 'mad_mes', 'median_mes']
        else:
            cols = ['fit_chisq', 'rminmes', 'rmesmed', 'rmesmad', 'rsnrmes', 'min_mes', 'mad_mes', 'median_mes']
            
        trainset.drop(cols, axis=1, inplace=True)
        testset.drop(cols, axis=1, inplace=True)
        valset.drop(cols, axis=1, inplace=True)
        
        self.trainset = trainset
        self.testset = testset
        self.valset = valset
        self.qt = qt
        

    def som_feature(self, sim, features, som, som_proportions):
        classify_array = np.load(self.input_loc / f'{sim}'/ f'{sim}_som_array{self.features_suffix}.npy')
        som_df = pd.read_csv(self.input_loc / f'{sim}'/ f'{sim}_som_ids{self.features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
        som_prob, som_dist = TransitSOM.classify(som, som_proportions, classify_array)
        
        som_df['SOM'] = som_prob
        som_df['SOM_dist'] = som_dist
        
        features = features.join(som_df)
        
        return features
    
    
    def load_trained_classified(self, clf_name):
        if clf_name == 'rfc':
            infile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_rfc.pkl'
            self.trained_rfc = utils.load_pickled_model(infile)
        elif clf_name == 'calibrated_rfc':
            infile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_rfc_calibrated.pkl'
            self.cal_rfc = utils.load_pickled_model(infile)
        elif clf_name == 'GP':
            infile = self.output / f'{self.sim1}-{self.sim2}_{self.model_suffix}_GP.pt'
            self.trained_GP = utils.load_trained_GP(infile)
        elif clf_name == 'XGB':
            infile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_XGB.json'
            self.trained_xgb = utils.load_trained_XGB(infile)    
        elif clf_name == 'rfc_XGB':
            infile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_rfc_XGB.json'
            self.trained_xgb = utils.load_trained_XGB(infile)
        
        
    def rfc(self, params, workers=1, save_model=True):
        clf = RandomForestClassifier(**params, n_jobs=workers, oob_score=True)
        self.trained_rfc = clf.fit(self.trainset, self.train_labels)
        
        if save_model:
            outfile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_rfc.pkl'
            utils.save_pickled_model(clf, outfile)
                
                
    def optimize_rfc(self, params):
        from sklearn.model_selection import GridSearchCV
        
        gscv = GridSearchCV(RandomForestClassifier(), param_grid=params, 
                            scoring=['accuracy', 'precision'], refit='accuracy',
                            verbose=2, n_jobs=12, cv=3)
        
        gscv.fit(self.trainset, self.train_labels)

        return gscv
                
    def calibrate_classifier(self, trained_clf, clf_name, save_model=True):            
        if len(self.valset) == 0:
            raise ValueError('Cannot calibrate a trained classifier without a validation set!')
                    
        self.calibrated = CalibratedClassifierCV(trained_clf, cv="prefit", method='isotonic')
        self.calibrated.fit(self.valset, self.val_labels)
        
        if save_model:
            outfile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_{clf_name}_calibrated.pkl'
            utils.save_pickled_model(outfile)
                
                
    def train_and_calibrate_rfc(self, params, fold_num=5, save_model=True):
        clf = RandomForestClassifier(**params)
        self.cal_rfc = CalibratedClassifierCV(clf, cv=fold_num, n_jobs=fold_num, method='isotonic', ensemble=True)
        self.cal_rfc.fit(self.trainset, self.train_labels)
        
        predictions = self.cal_rfc.predict(self.testset)
        prob = self.cal_rfc.predict_proba(self.testset)
        
        test_labels = self.test_labels.to_numpy()
        auc = roc_auc_score(test_labels, prob[:,1])
        acc = accuracy_score(test_labels, predictions)
        bl = brier_score_loss(test_labels, prob[:,1])
        ll = log_loss(test_labels, prob[:,1])
        
        print(f'AUC: {auc} ACC: {acc} BL: {bl} LL:{ll}')
        
        if save_model == True:
            outfile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}_rfc_calibrated.pkl'
            utils.save_pickled_model(self.cal_rfc, outfile)
                
    
    def train_GP(self, inducing_size, batch_size, epochs, early_stopping=True, save_model=True):
        if len(self.valset) == 0:
            raise ValueError('A validation set is required for training the GP!')
        
        X_train = self.trainset.copy()
        y_train = self.train_labels.to_numpy()
        
        X_valid = self.valset.to_numpy().astype(np.float32)
        y_valid = self.val_labels.to_numpy()
        
        valid_set = Dataset(X_valid, y_valid)

        half_ind = int(inducing_size/2)   
        missing_values = pd.isna(X_train).any(axis=1)
        X_sim1 = X_train.loc[~missing_values].query(f'sim == "{self.sim1}"').to_numpy()
        X_sim2 = X_train.loc[~missing_values].query(f'sim == "{self.sim2}"').to_numpy()
        km_0 = KMeans(half_ind).fit(X_sim1).cluster_centers_
        km_1 = KMeans(half_ind).fit(X_sim2).cluster_centers_

        X_inducing = np.concatenate([km_0, km_1])

        X_inducing = shuffle(X_inducing)

        X_inducing = X_inducing.astype(np.float32)

        X_inducing = torch.from_numpy(X_inducing)
    
        X_train, y_train = torch.from_numpy(X_train.to_numpy().astype(np.float32)), torch.from_numpy(y_train)
        num_training_samples = len(X_train)

        callbacks = []
        if early_stopping:
            es = EarlyStopping(monitor='valid_loss', patience=20, load_best=True, lower_is_better=True)
            callbacks.append(es)
            
        logging = PrintLog()
        callbacks.append(logging)

        gpclass = GPBinaryClassifier(
            GPC.VariationalModule,
            module__inducing_points=X_inducing,
            criterion__num_data=num_training_samples,
            device='cuda', max_epochs=epochs,
            optimizer=torch.optim.Adam,
            lr=0.01,
            batch_size=batch_size,
            train_split=predefined_split(valid_set),
            callbacks=[callbacks])

        gpclass.fit(np.array(X_train), np.array(y_train))

        self.trained_gp = gpclass
        
        if save_model:
            outfile = self.output / f'{self.sim1}-{self.sim2}_{self.model_suffix}_GP.pt'
            torch.save(gpclass.module_.state_dict(), outfile)
            
            model_params = gpclass.get_params(deep=True)
            model_params = {key: model_params[key] for key in ['module__inducing_points', 'criterion__num_data', 'batch_size', 'max_epochs', 'lr']}

            outfile = self.output / f'{self.sim1}-{self.sim2}_{self.model_suffix}_GP_params.pt'
            torch.save(model_params, outfile)
            
    
    def train_XGB(self, n_estimators=8000, max_depth=6, subsample=1.0, node_sample=0.5, tree_sample=0.5, lr=0.1, device='cuda',  early_stopping=True, rfc=False, save_model=True):
        callabks = []
        if early_stopping:
            es = xgb.callback.EarlyStopping(
                                        rounds=20,
                                        metric_name='logloss',
                                        data_name="validation_0", min_delta=1e-5,
                                        save_best=True)
            callabks.append(es)

        if rfc:
            num_parallel_tree = 100
            suffix = '_rfc'    
        else:
            num_parallel_tree = 1
            suffix = ''
            
        model = xgb.XGBClassifier(n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    subsample=subsample, 
                    colsample_bynode=node_sample, 
                    colsample_bytree=tree_sample, 
                    num_parallel_tree=num_parallel_tree, 
                    learning_rate=lr, 
                    objective='binary:logistic', 
                    device=device, 
                    eval_metric=['logloss'], 
                    callbacks=callabks)

        model.fit(self.trainset, self.train_labels.values,
            eval_set=[(self.valset, self.val_labels.values)], verbose=True)

        self.trained_xgb = model
        
        if save_model:
            outfile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}{suffix}_XGB.json'
            model.get_booster().save_model(outfile)
            
            params = model.get_xgb_params()
            outfile = self.output / f'{self.sim1}-{self.sim2}{self.model_suffix}{suffix}_XGB_params.json'
            with open(outfile, 'w') as f:
                json.dump(params, f)

                    
        
def select_training_ids(sim, input_loc='default', features_suffix='', train_size=0.9, train_suffix='training', val_size=0.0, val_suffix='validation', load_suffix=None):
    if input_loc == 'default':
        input_loc = Path(__file__).resolve().parents[1] / 'TrainingSet' / 'Simulations' 
    else:
        input_loc = Path(input_loc) 
            
    features = pd.read_csv(input_loc / f'{sim}' / f'{sim}_features{features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
    
    try:
        outliers = pd.read_csv(input_loc / f'{sim}' / f'{sim}_outliers{features_suffix}.csv').set_index(['sim_batch', 'sim_num'])
        
        features = features.loc[features.index.difference(outliers.index)]
    except FileNotFoundError:
        print('No outliers file found. Will assume that no outliers are present in training set.')
    
    if load_suffix is not None:
        existing_ids = pd.read_csv(input_loc / f'{sim}' / f'{sim}_{load_suffix}_ids{features_suffix}.csv').set_index(['sim_batch', 'sim_num']).index
    else:
        existing_ids = []
    
    if train_size <= 1.0:
        train_size = int(len(features)*train_size)
        
    if train_size > len(features):
        raise ValueError('Number of simulation features less than requested training set size!')
    
    if len(existing_ids) >= train_size:
        raise ValueError('Number of existing training set size greater or equal to requested!')
    
    if len(existing_ids) > 0:
        features.drop(existing_ids, inplace=True)
        train_size -= len(existing_ids)
    
        trainset = features.sample(train_size)
        
        new_ids = existing_ids.append(trainset.index)
        pd.DataFrame(index=new_ids).to_csv(input_loc / f'{sim}' / f'{sim}_{train_suffix}_ids{features_suffix}.csv')
    else:
        trainset = features.sample(train_size)
        pd.DataFrame(index=trainset.index).to_csv(input_loc / f'{sim}' / f'{sim}_{train_suffix}_ids{features_suffix}.csv')
    
    if val_size > 0:
        if val_size <= 1:
            val_size = int(len(features)*val_size)
        
        val_features = features.drop(trainset.index)
        
        if val_size > len(val_features):
            raise ValueError('Requested validation size greater than non-training sample!')
        
        val_features = val_features.sample(val_size)
        
        pd.DataFrame(index=val_features.index).to_csv(input_loc / f'{sim}' / f'{sim}_{val_suffix}_ids{features_suffix}.csv')
    
    
    
    