#!/us/bin/env python
#-*- coding:utf-8 -*-
##
## xgbooster.py
##
##  Created on: Dec 7, 2018
##      Author: Nina Narodytska, Alexey Ignatiev
##      E-mail: narodytska@vmware.com, aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
from .validate import SMTValidator
from .encode import SMTEncoder
from .explain import SMTExplainer
import numpy as np
import os
import resource
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

from  sklearn.preprocessing import OneHotEncoder
import sys
from six.moves import range
from .tree import TreeEnsemble
import xgboost as xgb
from xgboost import XGBClassifier, Booster
import pickle


#
#==============================================================================
class XGBooster(object):
    """
        The main class to train/encode/explain XGBoost models.
    """

    def __init__(self, options, from_data=None, from_model=None,
            from_encoding=None):
        """
            Constructor.
        """

        assert from_data or from_model or from_encoding, \
                'At least one input file should be specified'

        self.init_stime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self.init_ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

        # saving command-line options
        self.options = options
        self.seed = self.options.seed
        np.random.seed(self.seed)

        if from_data:
            self.use_categorical = self.options.use_categorical
            # saving data
            self.data = from_data
            dataset = np.asarray(self.data.samps, dtype=np.float32)


            # split data into X and y
            self.feature_names = self.data.names[:-1]
            self.nb_features = len(self.feature_names)

            self.X = dataset[:, 0:self.nb_features]
            self.Y = dataset[:, self.nb_features]
            self.num_class = len(set(self.Y))
            self.target_name = list(range(self.num_class))

            param_dist = {'n_estimators':self.options.n_estimators,
                      'max_depth':self.options.maxdepth}

            if(self.num_class == 2):
                param_dist['objective'] = 'binary:logistic'

            self.model = XGBClassifier(**param_dist)

            # split data into train and test sets
            self.test_size = self.options.testsplit
            if (self.test_size > 0):
                self.X_train, self.X_test, self.Y_train, self.Y_test = \
                        train_test_split(self.X, self.Y, test_size=self.test_size,
                                random_state=self.seed)
            else:
                self.X_train = self.X
                self.X_test = [] # need a fix
                self.Y_train = self.Y
                self.Y_test = []# need a fix

            # check if we have info about categorical features
            if (self.use_categorical):
                self.categorical_features = from_data.categorical_features
                self.categorical_names = from_data.categorical_names
                self.target_name = from_data.class_names

                ####################################
                # this is a set of checks to make sure that we use the same as anchor encoding
                cat_names = sorted(self.categorical_names.keys())
                assert(cat_names == self.categorical_features)
                self.encoder = {}
                for i in self.categorical_features:
                    self.encoder.update({i: OneHotEncoder(categories='auto', sparse=False)})#,
                    self.encoder[i].fit(self.X[:,[i]])

            else:
                self.categorical_features = []
                self.categorical_names = []
                self.encoder = []

            fname = from_data

        elif from_model:
            fname = from_model
            self.load_datainfo(from_model)
            if (self.use_categorical is False) and (self.options.use_categorical is True):
                print("Error: Note that the model is trained without categorical features info. Please do not use -c option for predictions")
                exit()
            # load model

        elif from_encoding:
            fname = from_encoding

            # encoding, feature names, and number of classes
            # are read from an input file
            enc = SMTEncoder(None, None, None, self, from_encoding)
            self.enc, self.intvs, self.imaps, self.ivars, self.feature_names, \
                    self.num_class = enc.access()

        # create extra file names
        try:
            os.stat(options.output)
        except:
            os.mkdir(options.output)

        self.mapping_features()
        #################
        self.test_encoding_transformes()

        bench_name = os.path.splitext(os.path.basename(options.files[0]))[0]
        bench_dir_name = options.output + "/" + bench_name
        try:
            os.stat(bench_dir_name)
        except:
            os.mkdir(bench_dir_name)

        self.basename = (os.path.join(bench_dir_name, bench_name +
                        "_nbestim_" + str(options.n_estimators) +
                        "_maxdepth_" + str(options.maxdepth) +
                        "_testsplit_" + str(options.testsplit)))

        data_suffix =  '.splitdata.pkl'
        self.modfile =  self.basename + '.mod.pkl'

        self.mod_plainfile =  self.basename + '.mod.txt'

        self.resfile =  self.basename + '.res.txt'
        self.encfile =  self.basename + '.enc.txt'
        self.expfile =  self.basename + '.exp.txt'

    def form_datefile_name(self, modfile):
        data_suffix =  '.splitdata.pkl'
        return  modfile + data_suffix

    def pickle_save_file(self, filename, data):
        try:
            f =  open(filename, "wb")
            pickle.dump(data, f)
            f.close()
        except:
            print("Cannot save to file", filename)
            exit()

    def pickle_load_file(self, filename):
        try:
            f =  open(filename, "rb")
            data = pickle.load(f)
            f.close()
            return data
        except:
            print("Cannot load from file", filename)
            exit()

    def save_datainfo(self, filename):

        print("saving  model to ", filename)
        self.pickle_save_file(filename, self.model)

        filename_data = self.form_datefile_name(filename)
        print("saving  data to ", filename_data)
        samples = {}
        samples["X"] = self.X
        samples["Y"] = self.Y
        samples["X_train"] = self.X_train
        samples["Y_train"] = self.Y_train
        samples["X_test"] = self.X_test
        samples["Y_test"] = self.Y_test
        samples["feature_names"] = self.feature_names
        samples["target_name"] = self.target_name
        samples["num_class"] = self.num_class
        samples["categorical_features"] = self.categorical_features
        samples["categorical_names"] = self.categorical_names
        samples["encoder"] = self.encoder
        samples["use_categorical"] = self.use_categorical


        self.pickle_save_file(filename_data, samples)

    def load_datainfo(self, filename):
        print("loading model from ", filename)
        self.model = XGBClassifier()
        self.model = self.pickle_load_file(filename)

        datafile = self.form_datefile_name(filename)
        print("loading data from ", datafile)
        loaded_data = self.pickle_load_file(datafile)
        self.X = loaded_data["X"]
        self.Y = loaded_data["Y"]
        self.X_train = loaded_data["X_train"]
        self.X_test = loaded_data["X_test"]
        self.Y_train = loaded_data["Y_train"]
        self.Y_test = loaded_data["Y_test"]
        self.feature_names = loaded_data["feature_names"]
        self.target_name = loaded_data["target_name"]
        self.num_class = loaded_data["num_class"]
        self.nb_features = len(self.feature_names)
        self.categorical_features = loaded_data["categorical_features"]
        self.categorical_names = loaded_data["categorical_names"]
        self.encoder = loaded_data["encoder"]
        self.use_categorical = loaded_data["use_categorical"]

    def train(self, outfile=None):
        """
            Train a tree ensemble using XGBoost.
        """

        return self.build_xgbtree(outfile)

    def encode(self, test_on=None):
        """
            Encode a tree ensemble trained previously.
        """

        encoder = SMTEncoder(self.model, self.feature_names, self.num_class, self)
        self.enc, self.intvs, self.imaps, self.ivars = encoder.encode()

        if test_on:
            encoder.test_sample(np.array(test_on))

        encoder.save_to(self.encfile)

    def explain(self, sample, use_lime=False, use_anchor=False, use_shap=False,
            expl_ext=None, prefer_ext=False, nof_feats=5):
        """
            Explain a prediction made for a given sample with a previously
            trained tree ensemble.
        """

        if use_lime:
            expl = use_lime(self, sample=sample, nb_samples=5,
                            nb_features_in_exp=nof_feats)
        elif use_anchor:
            expl = use_anchor(self, sample=sample, nb_samples=5,
                            nb_features_in_exp=nof_feats, threshold=0.95)
        elif use_shap:
            expl = use_shap(self, sample=sample, nb_features_in_exp=nof_feats)
        else:
            if 'x' not in dir(self):
                self.x = SMTExplainer(self.enc, self.intvs, self.imaps,
                        self.ivars, self.feature_names, self.num_class,
                        self.options, self)

            expl = self.x.explain(np.array(sample), self.options.smallest,
                    expl_ext, prefer_ext)

        # returning the explanation
        return expl

    def validate(self, sample, expl):
        """
            Make an attempt to show that a given explanation is optimistic.
        """

        # there must exist an encoding
        if 'enc' not in dir(self):
            encoder = SMTEncoder(self.model, self.feature_names, self.num_class,
                    self)
            self.enc, _, _, _ = encoder.encode()

        if 'v' not in dir(self):
            self.v = SMTValidator(self.enc, self.feature_names, self.num_class,
                    self)

        # try to compute a counterexample
        return self.v.validate(np.array(sample), expl)

    def transform(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.encoder != [])
            tx = []
            for i in range(self.nb_features):
                self.encoder[i].drop = None
                if (i in self.categorical_features):
                    tx_aux = self.encoder[i].transform(x[:,[i]])
                    tx_aux = np.vstack(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_inverse(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.encoder != [])
            inverse_x = []
            for i, xi in enumerate(x):
                inverse_xi = np.zeros(self.nb_features)
                for f in range(self.nb_features):
                    if f in self.categorical_features:
                        nb_values = len(self.categorical_names[f])
                        v = xi[:nb_values]
                        v = np.expand_dims(v, axis=0)
                        iv = self.encoder[f].inverse_transform(v)
                        inverse_xi[f] =iv
                        xi = xi[nb_values:]

                    else:
                        inverse_xi[f] = xi[0]
                        xi = xi[1:]
                inverse_x.append(inverse_xi)
            return inverse_x
        else:
            return x

    def transform_inverse_by_index(self, idx):
        if (idx in self.extended_feature_names):
            return self.extended_feature_names[idx]
        else:
            print("Warning there is no feature {} in the internal mapping".format(idx))
            return None

    def transform_by_value(self, feat_value_pair):
        if (feat_value_pair in self.extended_feature_names.values()):
            keys = (list(self.extended_feature_names.keys())[list( self.extended_feature_names.values()).index(feat_value_pair)])
            return keys
        else:
            print("Warning there is no value {} in the internal mapping".format(feat_value_pair))
            return None

    def mapping_features(self):
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        if (self.use_categorical):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.encoder[i].categories_[0]):
                        self.extended_feature_names.update({counter:  (self.feature_names[i], j)})
                        self.extended_feature_names_as_array_strings.append("f{}_{}".format(i,j)) # str(self.feature_names[i]), j))
                        counter = counter + 1
                else:
                    self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                    self.extended_feature_names_as_array_strings.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)

    def test_encoding_transformes(self):
        # test encoding

        X = self.X_train[[0],:]

        print("Sample of length", len(X[0])," : ", X)
        enc_X = self.transform(X)
        print("Encoded sample of length", len(enc_X[0])," : ", enc_X)
        inv_X = self.transform_inverse(enc_X)
        print("Back to sample", inv_X)
        print("Readable sample", self.readable_sample(inv_X[0]))
        assert((inv_X == X).all())

        if (self.options.verb > 1):
            for i in range(len(self.extended_feature_names)):
                print(i, self.transform_inverse_by_index(i))
            for key, value in self.extended_feature_names.items():
                print(value, self.transform_by_value(value))

    def transfomed_sample_info(self, i):
        print(enc.categories_)

    def build_xgbtree(self, outfile=None):
        """
            Build an ensemble of trees.
        """

        if (outfile is None):
            outfile = self.modfile
        else:
            self.datafile = sefl.form_datefile_name(outfile)

        # fit model no training data

        if (len(self.X_test) > 0):
            eval_set=[(self.transform(self.X_train), self.Y_train), (self.transform(self.X_test), self.Y_test)]
        else:
            eval_set=[(self.transform(self.X_train), self.Y_train)]

        print("start xgb")
        self.model.fit(self.transform(self.X_train), self.Y_train,
                  eval_set=eval_set,
                  verbose=self.options.verb) # eval_set=[(X_test, Y_test)],
        print("end xgb")

        evals_result =  self.model.evals_result()
        ########## saving model
        self.save_datainfo(outfile)
        print("saving plain model to ", self.mod_plainfile)
        self.model._Booster.dump_model(self.mod_plainfile)

        ensemble = TreeEnsemble(self.model, self.extended_feature_names_as_array_strings, nb_classes = self.num_class)

        y_pred_prob = self.model.predict_proba(self.transform(self.X_train[:10]))
        y_pred_prob_compute = ensemble.predict(self.transform(self.X_train[:10]), self.num_class)

        assert(np.absolute(y_pred_prob_compute- y_pred_prob).sum() < 0.01*len(y_pred_prob))

        ### accuracy
        try:
            train_accuracy = round(1 - evals_result['validation_0']['merror'][-1],2)
        except:
            try:
                train_accuracy = round(1 - evals_result['validation_0']['error'][-1],2)
            except:
                assert(False)

        try:
            test_accuracy = round(1 - evals_result['validation_1']['merror'][-1],2)
        except:
            try:
                test_accuracy = round(1 - evals_result['validation_1']['error'][-1],2)
            except:
                print("no results test data")
                test_accuracy = 0


        #### saving
        print("saving results to ", self.resfile)
        with open(self.resfile, 'w') as f:
            f.write("{} & {} & {} &{}  &{} & {} \\\\ \n \hline \n".format(
                                           os.path.basename(self.options.files[0]).replace("_","-"),
                                           train_accuracy,
                                           test_accuracy,
                                           self.options.n_estimators,
                                           self.options.maxdepth,
                                           self.options.testsplit))
        f.close()

        print("Train accuracy: %.2f%%" % (train_accuracy * 100.0))
        print("Test accuracy: %.2f%%" % (test_accuracy * 100.0))


        return train_accuracy, test_accuracy, self.model
