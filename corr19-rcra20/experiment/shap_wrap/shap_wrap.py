#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## shap_wrap.py (reuses parts of the code of SHAP)
##
##  Created on: Sep 25, 2019
##      Author: Nina Narodytska
##      E-mail: narodytska@vmware.com
##

#
#==============================================================================
import json
import numpy as np
import xgboost as xgb
import math
import shap
import resource


#
#==============================================================================
def shap_call(xgb, sample = None, feats='all', nb_features_in_exp = None):
    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    f2imap = {}
    for i, f in enumerate(xgb.feature_names):
        f2imap[f.strip()] = i

    if (sample is not None):
        if (nb_features_in_exp is None):
            nb_features_in_exp = len(sample)

        try:
            feat_sample  = np.asarray(sample, dtype=np.float32)
        except:
            print("Cannot parse input sample:", sample)
            exit()
        print("\n\n Starting SHAP explainer... \n Considering a sample with features:", feat_sample)
        if not (len(feat_sample) == len(xgb.X_train[0])):
            print("Unmatched features are not supported: The number of features in a sample {} is not equal to the number of features in this benchmark {}".format(len(feat_sample), len(xgb.X_train[0])))
            exit()

        # compute boost predictions
        feat_sample_exp = np.expand_dims(feat_sample, axis=0)
        feat_sample_exp = xgb.transform(feat_sample_exp)
        y_pred = xgb.model.predict(feat_sample_exp)[0]
        y_pred_prob = xgb.model.predict_proba(feat_sample_exp)[0]

        # No need to pass dataset as it is recored in model
        # https://shap.readthedocs.io/en/latest/

        explainer = shap.TreeExplainer(xgb.model)
        shap_values = explainer.shap_values(feat_sample_exp)

        shap_values_sample = shap_values[-1]
        transformed_sample = feat_sample_exp[-1]




        # we need to sum values per feature
        # https://github.com/slundberg/shap/issues/397
        sum_values = []
        if (xgb.use_categorical):
            p = 0
            for f in xgb.categorical_features:
                nb_values = len(xgb.categorical_names[f])
                sum_v = 0
                for i in range(nb_values):
                    sum_v = sum_v + shap_values_sample[p+i]
                p = p + nb_values
                sum_values.append(sum_v)
        else:
            sum_values = shap_values_sample
        expl = []

        # choose which features in the explanation to focus on
        if feats in ('p', 'pos', '+'):
            feats = 1
        elif feats in ('n', 'neg', '-'):
            feats = -1
        else:
            feats = 0

        print("\t \t Explanations for the winner class", y_pred, " (xgboost confidence = ", y_pred_prob[int(y_pred)], ")")
        print("base_value = {}, predicted_value = {}".format(explainer.expected_value, np.sum(sum_values) + explainer.expected_value))

        abs_sum_values = np.abs(sum_values)
        sorted_by_abs_sum_values =np.argsort(-abs_sum_values)

        for k1, v1 in enumerate(sorted_by_abs_sum_values):

            k = v1
            v = sum_values[v1]

            if (feats == 1 and v < 0) or (feats == -1 and v >= 0):
                continue

            expl.append(f2imap[xgb.feature_names[k]])
            print("id = {}, name = {}, score = {}".format(f2imap[xgb.feature_names[k]], xgb.feature_names[k], v))

            if (len(expl) ==  nb_features_in_exp):
                break

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        print('  time: {0:.2f}'.format(timer))

        return sorted(expl[:nb_features_in_exp])
