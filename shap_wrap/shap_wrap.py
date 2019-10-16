"""
CODE REUSES FROM SHAP
"""
import json
import numpy as np
import xgboost as xgb
import math
import shap
import resource


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
        
        #X_train_clone = np.vstack((xgb.X_train, feat_sample))
        #transformed_train =  xgb.transform(X_train_clone)
        
        explainer = shap.TreeExplainer(xgb.model)
        shap_values = explainer.shap_values(feat_sample_exp)
        #print(explainer.expected_value)
        #exit()
        
        shap_values_sample = shap_values[-1]
        transformed_sample = feat_sample_exp[-1] #transformed_train[-1]


        

        # we need to sum values per feature 
        # https://github.com/slundberg/shap/issues/397
        #print(xgb.categorical_features, len(shap_values), len(transformed_sample))
        sum_values = []
        if (xgb.use_categorical):
            p = 0
            #print(xgb.categorical_features)
            for f in xgb.categorical_features:
                #print(xgb.categorical_names[f])
                nb_values = len(xgb.categorical_names[f])
                sum_v = 0
                #print(nb_values)
                for i in range(nb_values):
                    #print(i, p+i)
                    sum_v = sum_v + shap_values_sample[p+i]
                p = p + nb_values
                #print(p, sum_v)
                sum_values.append(sum_v)
            #print(sum_values)
        else:
            sum_values = shap_values_sample
        expl_for_sampling = [{"base_value": explainer.expected_value, "predicted_value":np.sum(sum_values) + explainer.expected_value}]
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
        #print(abs_sum_values)
        sorted_by_abs_sum_values =np.argsort(-abs_sum_values)
        #print(sorted_by_abs_sum_values)

        
        for k1, v1 in enumerate(sorted_by_abs_sum_values):
            
            k = v1
            v = sum_values[v1]
            
            if (feats == 1 and v < 0) or (feats == -1 and v >= 0):
                continue                      
            #expl.append(v)
            #print(feat_sample[k])
            if (xgb.use_categorical):
                expl_for_sampling.append(
                    [{"id":k, "score": v, "name":"", "value": feat_sample[k],  "original_name": xgb.feature_names[k], "original_value": xgb.categorical_names[k][int(feat_sample[k])]}])
            else:
                expl_for_sampling.append(
                    [{"id":k, "score": v, "name":"", "value": feat_sample[k],  "original_name": xgb.feature_names[k], "original_value": feat_sample[k]}])
            expl.append(f2imap[xgb.feature_names[k]])
            print("id = {}, name = {}, score = {}".format(f2imap[xgb.feature_names[k]], xgb.feature_names[k], v))
            
            #print(len(expl),nb_features_in_exp)
            if (len(expl) ==  nb_features_in_exp):
                break

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        print('  time: {0:.2f}'.format(timer))

        #print(expl_for_sampling)
        return sorted(expl[:nb_features_in_exp]), expl_for_sampling

            
# 
#     
#     
#     max_sample = 10
#     y_pred_prob = xgb.model.predict_proba(xgb.X_test)
#     y_pred = xgb.model.predict(xgb.X_test)
# 
#     nb_tests = min(max_sample,len(xgb.y_test))
#     top_labels = 1
#     for sample in range(nb_tests):
#         np.set_printoptions(precision=2)
#         feat_sample = xgb.X_test[sample]
#         print("Considering a sample with features:", feat_sample)        
#         if (False):            
#             feat_sample[4] = 3000
#             #feat_sample[3] = 100
#             y_pred_prob_sample = xgb.model.predict_proba([feat_sample])
#             print(y_pred_prob_sample)
#         print("\t Predictions:", y_pred_prob[sample])
#         
#         
#         if (xgb.num_class  > 2):
#             for i in range(xgb.num_class):               
#                 print("\t \t Explanations for class", i, " (xgboost confidence = ", y_pred_prob[sample][i], ")")
#                 print("\t \t ", shap_values[i][sample])
#         else:
#             i = int(y_pred[sample])
#             print("\t \t Explanations for class", i, " (xgboost confidence = ", y_pred_prob[sample][i], ")")
#             print("\t \t Imact of features", shap_values[sample])
#             
#     #exp.save_to_file("1.html")
#     #plt.savefig("y", dpi=72)