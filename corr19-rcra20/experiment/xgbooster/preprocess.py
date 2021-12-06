#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## preprocess.py
##
##  Created on: Jan 10, 2019
##      Author: Nina Narodytska
##      E-mail: narodytska@vmware.com
##

#
#==============================================================================
import json
import numpy as np
import xgboost as xgb
import math
import pandas as pd
import numpy as np
import sklearn
import pickle


#
#==============================================================================
def preprocess_dataset(raw_data_path, files):
    print("preprocess dataset from ", raw_data_path)
    files = files.split(",")

    data_file = files[0]
    dataset_name = files[1]

    try:
        data_raw = pd.read_csv(raw_data_path + data_file, sep=',', na_values=  [''])
        catcols = pd.read_csv(raw_data_path + data_file + ".catcol", header = None)
        categorical_features = np.concatenate(catcols.values).tolist()


        for i in range(len(data_raw.values[0])):
            if i in categorical_features:
                data_raw.fillna('',inplace=True)
            else:
                data_raw.fillna(0,inplace=True)
        dataset_all = data_raw
        dataset =  dataset_all.values.copy()

        print(categorical_features)
    except Exception as e:
        print("Please provide info about categorical columns/original datasets or omit option -p", e)
        exit()

    # move categrorical columns forward

    feature_names = dataset_all.columns
    print(feature_names)

    ##############################
    extra_info = {}
    categorical_names = {}
    print(categorical_features)
    dataset_new = dataset_all.values.copy()
    for feature in categorical_features:
        print("feature", feature)
        print(dataset[:, feature])
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(dataset[:, feature])
        categorical_names[feature] = le.classes_
        dataset_new[:, feature] = le.transform(dataset[:, feature])

    ###################################3
    # target as categorical
    labels_new = []

    le = sklearn.preprocessing.LabelEncoder()
    le.fit(dataset[:, -1])
    dataset_new[:, -1]= le.transform(dataset[:, -1])
    class_names = le.classes_
    ######################################33


    if (False):
        dataset_new = np.delete(dataset_new, -1, axis=1)
        oneencoder = sklearn.preprocessing.OneHotEncoder()
        oneencoder.fit(dataset_new[:, categorical_features])
        print(oneencoder.categories_)
        n_transformed_features = sum([len(cats) for cats in oneencoder.categories_])
        print(n_transformed_features)
        print(dataset_new.shape)
        X = dataset_new[:,categorical_features][0]
        print(X)
        x = np.expand_dims(X, axis=0)
        print("x", x, x.shape)
        y = dataset_new[0].copy()
        print(y.shape, oneencoder.transform(x).shape)
        y[categorical_features] = oneencoder.transform(x).toarray()

        print("y", y, y.shape)

        z = oneencoder.inverse_transform(y)
        print(z.shape)
        exit()

    ###########################################################################3
    extra_info = {"categorical_features": categorical_features,
                  "categorical_names": categorical_names,
                  "feature_names": feature_names,
                  "class_names": class_names}

    new_file_train = raw_data_path + dataset_name + '_data.csv'
    df = pd.DataFrame(data=dataset_new)
    df.columns = list(feature_names)
    df.to_csv(new_file_train, mode = 'w', index=False)
    print("new dataset", new_file_train)


    f =  open(raw_data_path + dataset_name + '_data.csv.pkl', "wb")
    pickle.dump(extra_info, f)
    f.close()
