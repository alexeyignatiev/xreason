#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev, Nina Narodytska
##      E-mail: aignatiev@ciencias.ulisboa.pt, narodytska@vmware.com
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
import sys


#
#==============================================================================
class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command):
        """
            Constructor.
        """

        # actions
        self.train = False
        self.encode = 'none'
        self.explain = ''
        self.useanchor = False
        self.uselime = False
        self.useshap = False
        self.limefeats = 5
        self.validate = False
        self.use_categorical = False
        self.preprocess_categorical = False
        self.preprocess_categorical_files = ""

        # training options
        self.accmin = 0.95
        self.n_estimators = 100
        self.num_boost_round = 10
        self.maxdepth = 3
        self.testsplit = 0.2
        self.seed = 7

        # other options
        self.files = None
        self.output = 'temp'
        self.mapfile = None
        self.separator = ','
        self.smallest = False
        self.solver = 'z3'
        self.verb = 0

        if command:
            self.parse(command)

    def parse(self, command):
        """
            Parser.
        """

        self.command = command

        try:
            opts, args = getopt.getopt(command[1:],
                                    'a:ce:d:hL:lm:Mn:o:pr:qs:tvVwx:',
                                    ['accmin=',
                                     'encode=',
                                     'help',
                                     'map-file=',
                                     'use-anchor=',
                                     'lime-feats=',
                                     'use-lime=',
                                     'use-shap=',
                                     'use-categorical=',
                                     'preprocess-categorical=',
                                     'pfiles=',
                                     'maxdepth=',
                                     'minimum',
                                     'nbestims=',
                                     'output=',
                                     'rounds='
                                     'seed=',
                                     'sep=',
                                     'solver=',
                                     'testsplit=',
                                     'train',
                                     'validate',
                                     'verbose',
                                     'explain='
                                     ])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-a', '--accmin'):
                self.accmin = float(arg)
            elif opt in ('-c', '--use-categorical'):
                self.use_categorical = True
            elif opt in ('-d', '--maxdepth'):
                self.maxdepth = int(arg)
            elif opt in ('-e', '--encode'):
                self.encode = str(arg)
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)
            elif opt in ('-l', '--use-lime'):
                self.uselime = True
            elif opt in ('-L', '--lime-feats'):
                self.limefeats = 0 if arg == 'all' else int(arg)
            elif opt in ('-m', '--map-file'):
                self.mapfile = str(arg)
            elif opt in ('-M', '--minimum'):
                self.smallest = True
            elif opt in ('-n', '--nbestims'):
                self.n_estimators = int(arg)
            elif opt in ('-o', '--output'):
                self.output = str(arg)
            elif opt in ('-q', '--use-anchor'):
                self.useanchor = True
            elif opt in ('-r', '--rounds'):
                self.num_boost_round = int(arg)
            elif opt == '--seed':
                self.seed = int(arg)
            elif opt == '--sep':
                self.separator = str(arg)
            elif opt in ('-s', '--solver'):
                self.solver = str(arg)
            elif opt == '--testsplit':
                self.testsplit = float(arg)
            elif opt in ('-t', '--train'):
                self.train = True
            elif opt in ('-V', '--validate'):
                self.validate = True
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt in ('-w', '--use-shap'):
                self.useshap = True
            elif opt in ('-x', '--explain'):
                self.explain = str(arg)
            elif opt in ('-p', '--preprocess-categorical'):
                self.preprocess_categorical = True
            elif opt in ('--pfiles'):
                self.preprocess_categorical_files = str(arg) #train_file, test_file(or empty, resulting file
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        if self.encode == 'none':
            self.encode = None

        self.files = args

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] input-file')
        print('Options:')
        print('        -a, --accmin=<float>       Minimal accuracy')
        print('                                   Available values: [0.0, 1.0] (default = 0.95)')
        print('        -c, --use-categorical      Treat categorical features as categorical (with categorical features info if available)')
        print('        -d, --maxdepth=<int>       Maximal depth of a tree')
        print('                                   Available values: [1, INT_MAX] (default = 3)')
        print('        -e, --encode=<smt>         Encode a previously trained model')
        print('                                   Available values: smt, smtbool, none (default = none)')
        print('        -h, --help                 Show this message')
        print('        -l, --use-lime             Use LIME to compute an explanation')
        print('        -L, --lime-feats           Instruct LIME to compute an explanation of this size')
        print('                                   Available values: [1, INT_MAX], all (default = 5)')
        print('        -m, --map-file=<string>    Path to a file containing a mapping to original feature values. (default: none)')
        print('        -M, --minimum              Compute a smallest size explanation (instead of a subset-minimal one)')
        print('        -n, --nbestims=<int>       Number of trees per class')
        print('                                   Available values: [1, INT_MAX] (default = 100)')
        print('        -o, --output=<string>      Directory where output files will be stored (default: \'temp\')')
        print('        -p,                        Preprocess categorical data')
        print('        --pfiles                   Filenames to use when preprocessing')
        print('        -q, --use-anchor           Use Anchor to compute an explanation')
        print('        -r, --rounds=<int>         Number of training rounds')
        print('                                   Available values: [1, INT_MAX] (default = 10)')
        print('        --seed=<int>               Seed for random splitting')
        print('                                   Available values: [1, INT_MAX] (default = 7)')
        print('        --sep=<string>             Field separator used in input file (default = \',\')')
        print('        -s, --solver=<string>      An SMT reasoner to use')
        print('                                   Available values: cvc4, mathsat, yices, z3 (default = z3)')
        print('        -t, --train                Train a model of a given dataset')
        print('        --testsplit=<float>        Training and test sets split')
        print('                                   Available values: [0.0, 1.0] (default = 0.2)')
        print('        -v, --verbose              Increase verbosity level')
        print('        -V, --validate             Validate explanation (show that it is too optimistic)')
        print('        -w, --use-shap             Use SHAP to compute an explanation')
        print('        -x, --explain=<string>     Explain a decision for a given comma-separated sample (default: none)')
