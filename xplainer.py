#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
from data import Data
from anchor_wrap import anchor_call
from lime_wrap import lime_call
from shap_wrap import shap_call
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset


def show_info():
    """
        Print info message.
    """

    print('c XPlainer: explaining XGBoost models')
    print('')


if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

    if (options.preprocess_categorical):
        preprocess_dataset(options.files[0], options.preprocess_categorical_files)
        exit()

    if options.files:
        xgb = None

        if options.train:
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)

            xgb = XGBooster(options, from_data=data)
            train_accuracy, test_accuracy, model = xgb.train()

        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]

        if options.encode:
            if not xgb:
                xgb = XGBooster(options, from_model=options.files[0])

            # encode it and save the encoding to another file
            xgb.encode(test_on=options.explain)
        if options.explain:
            if not xgb:
                if options.uselime or options.useanchor or options.useshap:
                    xgb = XGBooster(options, from_model=options.files[0])
                else:
                    # abduction-based approach requires an encoding
                    xgb = XGBooster(options, from_encoding=options.files[0])
            # checking LIME should use all features
            if not options.limefeats:
                options.limefeats = len(data.names) - 1

            # explain using anchor or the abduction-based approach
            expl, expl_for_sampling = xgb.explain(options.explain,
                    use_lime=lime_call if options.uselime else None,
                    use_anchor=anchor_call if options.useanchor else None,
                    use_shap=shap_call if options.useshap else None,)

            if options.yieldexp is not 'none':
                print("\n\nExplainer output:")
                for v in expl_for_sampling: 
                    print(v)
                exit()
     
            if (options.uselime or options.useanchor) and options.validate:
                xgb.validate(options.explain, expl)

