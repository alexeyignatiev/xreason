"""
CODE REUSES FROM SHAP
"""
import json
import numpy as np
import xgboost as xgb
import math
import lime
import lime.lime_tabular
import resource


def lime_call(xgb, sample = None, nb_samples = 5, feats='all',
        nb_features_in_exp=5):

    timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
            resource.getrusage(resource.RUSAGE_SELF).ru_utime

    # we need a way to say that features are categorical ?
    # we do not have this informations.
    predict_fn_xgb = lambda x: xgb.model.predict_proba(xgb.transform(x)).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        xgb.X_train,
        feature_names=xgb.feature_names,
        categorical_features=xgb.categorical_features if xgb.use_categorical else None,
        class_names=xgb.target_name,
        discretize_continuous=True,
        )

    f2imap = {}
    for i, f in enumerate(xgb.feature_names):
        f2imap[f.strip()] = i

    if (sample is not None):
        try:
            feat_sample  = np.asarray(sample, dtype=np.float32)
        except:
            print("Cannot parse input sample:", sample)
            exit()
        print("Considering a sample with features:", feat_sample)
        if not (len(feat_sample) == len(xgb.X_train[0])):
            print("Unmatched features are not supported: The number of features in a sample {} is not equal to the number of features in this benchmark {}".format(len(feat_sample), len(xgb.X_train[0])))
            exit()

        # compute boost predictions
        feat_sample_exp = np.expand_dims(feat_sample, axis=0)
        feat_sample_exp = xgb.transform(feat_sample_exp)
        y_pred = xgb.model.predict(feat_sample_exp)[0]
        y_pred_prob = xgb.model.predict_proba(feat_sample_exp)[0]

        exp = explainer.explain_instance(feat_sample,
                                         predict_fn_xgb,
                                         num_features = nb_features_in_exp,
                                         top_labels = 1)#,
                                         #labels = list(range(xgb.num_class)))



        # we collect info for the top class 
        expl_for_sampling = []
        expl = []

        # choose which features in the explanation to focus on
        if feats in ('p', 'pos', '+'):
            feats = 1
        elif feats in ('n', 'neg', '-'):
            feats = -1
        else:
            feats = 0

        for i in range(xgb.num_class):
            if (i !=  y_pred):
                continue
            print("\t \t Explanations for the winner class", i, " (xgboost confidence = ", y_pred_prob[i], ")")
            print("\t \t Features in explanations: ", exp.as_list(label=i))

            s_human_readable = ""
            for k, v in enumerate(exp.as_list(label=i)):
                s_human_readable_k = ""
                name = ""
                value = ""
                if (feats == 1 and v[1] < 0) or (feats == -1 and v[1] >= 0):
                    continue

                if not (('<'  in  v[0]) or  ('>'  in  v[0])):
                    a = v[0].split('=')
                    f = a[0].strip()
                    l = a[1].strip()
                    u = l
                    name = f 
                    if (xgb.use_categorical):
                        fid =  f2imap[f]
                        fvid = int(a[1])
                        s_human_readable_k =   f  + " = [" + str(xgb.categorical_names[fid][fvid]) +"," + str(v[1])+ "] "
                        s_human_readable = s_human_readable + s_human_readable_k
                        value = str(xgb.categorical_names[fid][fvid]) 

                else:
                    a = v[0].split('<')

                    if len(a) == 1:
                        a = v[0].split('>')

                    if len(a) == 2:
                        f = a[0].strip()

                        if '>' in v[0]:
                            l, u = float(a[1].strip(' =')), None
                        else:
                            l, u = None, float(a[1].strip(' ='))
                    else:
                        l = float(a[0].strip())
                        f = a[1].strip(' =')
                        u = float(a[2].strip(' ='))
                    name = f
                # expl.append(tuple([f2imap[f], l, u, v[1] >= 0]))
                expl.append(f2imap[f])
                if (xgb.use_categorical):
                    assert(l == u)
                    expl_for_sampling.append([{"id":f2imap[f], "score":v[1], "name":v[0], "values":float(l),  "original_name":f, "original_value":value}])
                else:
                    expl_for_sampling.append([{"id":f2imap[f], "score":v[1], "name":v[0], "values":[l,u],  "original_name":f, "original_value": "NA"}])
            if (xgb.use_categorical):
                if (len(s_human_readable) > 0):
                    print("\t \t Features in explanations (with provided categorical labels): ", s_human_readable)

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        print('  time: {0:.2f}'.format(timer))
        #print(expl_for_sampling)
        return sorted(expl), expl_for_sampling

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
