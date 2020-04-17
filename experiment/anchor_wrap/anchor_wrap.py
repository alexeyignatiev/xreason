#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## anchor_wrap.py (reuses parts of the code of SHAP)
##
##  Created on: Jan 6, 2019
##      Author: Nina Narodytska, Alexey Ignatiev
##      E-mail: narodytska@vmware.com, aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import json
import numpy as np
import xgboost as xgb
import math
import resource
from anchor import utils
from anchor import anchor_tabular
import sklearn
import sklearn.ensemble


#
#==============================================================================
def anchor_call(xgb, sample=None, nb_samples=5, feats='all',
        nb_features_in_exp=5, threshold=0.95):

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    # we need a way to say that features are categorical ?
    # we do not have this informations.
    explainer = anchor_tabular.AnchorTabularExplainer(
                                                     class_names=xgb.target_name,
                                                     feature_names=xgb.feature_names,
                                                     train_data=xgb.X,
                                                     categorical_names=xgb.categorical_names if xgb.use_categorical else {})
    # if (len(xgb.X_test) != 0):
    #     explainer.fit(xgb.X_train, xgb.Y_train, xgb.X_test, xgb.Y_test)
    # else:
    #     explainer.fit(xgb.X_train, xgb.Y_train, xgb.X_train, xgb.Y_train)
    predict_fn_xgb = lambda x: xgb.model.predict(xgb.transform(x)).astype(int)

    f2imap = {}
    for i, f in enumerate(xgb.feature_names):
        f2imap[f.strip()] = i

    if (sample is not None):
        try:
            feat_sample = np.asarray(sample, dtype=np.float32)
        except Exception as inst:
            print("Cannot parse input sample:", sample, inst)
            exit()
        print("\n\n\n Starting Anchor explainer... \nConsidering a sample with features:", feat_sample)
        if not (len(feat_sample) == len(xgb.X_train[0])):
            print("Unmatched features are not supported: The number of features in a sample {} is not equal to the number of features in this benchmark {}".format(len(feat_sample), len(xgb.X_train[0])))
            exit()

        # compute boost predictions
        feat_sample_exp = np.expand_dims(feat_sample, axis=0)
        feat_sample_exp = xgb.transform(feat_sample_exp)
        y_pred = xgb.model.predict(feat_sample_exp)[0]
        y_pred_prob = xgb.model.predict_proba(feat_sample_exp)[0]
        #hack testiing that we use the same onehot encoding
        # test_feat_sample_exp = explainer.encoder.transform(feat_sample_exp)
        test_y_pred = xgb.model.predict(feat_sample_exp)[0]
        test_y_pred_prob = xgb.model.predict_proba(feat_sample_exp)[0]
        assert(np.allclose(y_pred_prob, test_y_pred_prob))
        print('Prediction: ', explainer.class_names[predict_fn_xgb(feat_sample.reshape(1, -1))[0]])
        # exp = explainer.explain_instance(feat_sample, xgb.model.predict, threshold=threshold)
        exp = explainer.explain_instance(feat_sample, predict_fn_xgb, threshold=threshold)
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())
        #print(exp.features())
        #print(exp.names())

        # explanation
        expl = []

        if (xgb.use_categorical):
            for k, v in enumerate(exp.features()):
                expl.append(v)
                print("Clause ", k, end=": ")
                print("feature (", v,  ",",  explainer.feature_names[v], end="); ")
                print("value (", feat_sample[v],  ",",  explainer.categorical_names[v][int(feat_sample[v])] , ")")
        else:
            print("We only support datasets with categorical features for Anchor. Please pre-process your data.")
            exit()

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        print('  time: {0:.2f}'.format(timer))

        return sorted(expl)

    ###################################### TESTING
    max_sample = nb_samples
    y_pred_prob = xgb.model.predict_proba(xgb.X_test)
    y_pred = xgb.model.predict(xgb.X_test)

    nb_tests = min(max_sample,len(xgb.Y_test))
    top_labels = 1
    for sample in range(nb_tests):
        np.set_printoptions(precision=2)
        feat_sample = xgb.X_test[sample]
        print("Considering a sample with features:", feat_sample)
        if (False):
            feat_sample[4] = 3000
            y_pred_prob_sample = xgb.model.predict_proba([feat_sample])
            print(y_pred_prob_sample)
            print("\t Predictions:", y_pred_prob[sample])
        exp = explainer.explain_instance(feat_sample,
                                         predict_fn_xgb,
                                         num_features= xgb.num_class,
                                         top_labels = 1,
                                         labels = list(range(xgb.num_class)))
        for i in range(xgb.num_class):
            if (i !=  y_pred[sample]):
                continue
            print("\t \t Explanations for the winner class", i, " (xgboost confidence = ", y_pred_prob[sample][i], ")")
            print("\t \t Features in explanations: ", exp.as_list(label=i))
    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
    print('  time: {0:.2f}'.format(timer))
    return
