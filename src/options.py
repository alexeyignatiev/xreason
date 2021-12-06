#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev, Nina Narodytska
##      E-mail: alexey.ignatiev@monash.edu, narodytska@vmware.com
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
from pysat.card import EncType
import sys

#
#==============================================================================
encmap = {
    "pw": EncType.pairwise,
    "seqc": EncType.seqcounter,
    "cardn": EncType.cardnetwrk,
    "sortn": EncType.sortnetwrk,
    "tot": EncType.totalizer,
    "mtot": EncType.mtotalizer,
    "kmtot": EncType.kmtotalizer,
    "native": EncType.native
}


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
        self.relax = 0
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

        # maxsat options
        self.minz = False
        self.am1 = False
        self.exhaust = False
        self.trim = 0

        # other options
        self.files = None
        self.cardenc = 'seqc'
        self.output = 'temp'
        self.mapfile = None
        self.reduce = 'none'
        self.separator = ','
        self.smallest = False
        self.solver = 'z3'
        self.unit_mcs = False
        self.verb = 0
        self.xnum = 1
        self.xtype = 'abd'

        if command:
            self.parse(command)

    def parse(self, command):
        """
            Parser.
        """

        self.command = command

        try:
            opts, args = getopt.getopt(command[1:],
                                    '1a:C:ce:Ed:hL:lm:Mn:N:o:pr:R:qs:tT:uvVwx:X:z',
                                    ['am1', 'accmin=', 'encode=', 'cardenc=',
                                     'exhaust', 'help', 'map-file=',
                                     'use-anchor=', 'lime-feats=', 'use-lime=',
                                     'use-shap=', 'use-categorical=',
                                     'preprocess-categorical=', 'pfiles=',
                                     'maxdepth=', 'minimum', 'nbestims=',
                                     'output=', 'reduce=', 'rounds=', 'relax=',
                                     'seed=', 'sep=', 'solver=', 'testsplit=',
                                     'train', 'trim=', 'unit-mcs', 'validate',
                                     'verbose', 'xnum=', 'xtype=', 'explain=',
                                     'minz'])
        except getopt.GetoptError as err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)

        for opt, arg in opts:
            if opt in ('-1', '--am1'):
                self.am1 = True
            elif opt in ('-a', '--accmin'):
                self.accmin = float(arg)
            elif opt in ('-c', '--use-categorical'):
                self.use_categorical = True
            elif opt in ('-C', '--cardenc'):
                self.cardenc = str(arg)
            elif opt in ('-d', '--maxdepth'):
                self.maxdepth = int(arg)
            elif opt in ('-e', '--encode'):
                self.encode = str(arg)
            elif opt in ('-E', '--exhaust'):
                self.exhaust = True
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
            elif opt in ('-N', '--xnum'):
                self.xnum = str(arg)
                self.xnum = -1 if self.xnum == 'all' else int(self.xnum)
            elif opt in ('-o', '--output'):
                self.output = str(arg)
            elif opt in ('-p', '--preprocess-categorical'):
                self.preprocess_categorical = True
            elif opt in ('--pfiles'):
                self.preprocess_categorical_files = str(arg) #train_file, test_file(or empty, resulting file
            elif opt in ('-q', '--use-anchor'):
                self.useanchor = True
            elif opt in ('-r', '--rounds'):
                self.num_boost_round = int(arg)
            elif opt in ('-R', '--reduce'):
                self.reduce = str(arg)
            elif opt == '--relax':
                self.relax = int(arg)
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
            elif opt in ('-T', '--trim'):
                self.trim = int(arg)
            elif opt in ('-u', '--unit-mcs'):
                self.unit_mcs = True
            elif opt in ('-V', '--validate'):
                self.validate = True
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt in ('-w', '--use-shap'):
                self.useshap = True
            elif opt in ('-x', '--explain'):
                self.explain = str(arg)
            elif opt in ('-X', '--xtype'):
                self.xtype = str(arg)
            elif opt in ('-z', '--minz'):
                self.minz = True
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        if self.encode == 'none':
            self.encode = None
        elif self.encode in ('mx', 'mxe', 'maxsat', 'mxint', 'mxa') and self.solver in ('cvc4', 'mathsat', 'yices', 'z3'):
            # setting the default solver for the mxreasoning-based oracle
            self.solver = 'm22'

        # assigning the encoding for cardinality constraints
        self.cardenc = encmap[self.cardenc]

        self.files = args

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] input-file')
        print('Options:')
        print('        -1, --am1                  Adapt AM1 constraints when running RC2')
        print('        -a, --accmin=<float>       Minimal accuracy')
        print('                                   Available values: [0.0, 1.0] (default = 0.95)')
        print('        -c, --use-categorical      Treat categorical features as categorical (with categorical features info if available)')
        print('        -C, --cardenc=<string>     Cardinality encoding to use')
        print('                                   Available values: cardn, kmtot, mtot, sortn, seqc, tot (default = seqc)')
        print('        -d, --maxdepth=<int>       Maximal depth of a tree')
        print('                                   Available values: [1, INT_MAX] (default = 3)')
        print('        -e, --encode=<string>      Encode a previously trained model')
        print('                                   Available values: maxsat, smt, smtbool, none (default = none)')
        print('        -E, --exhaust              Apply core exhaustion when running RC2')
        print('        -h, --help                 Show this message')
        print('        -l, --use-lime             Use LIME to compute an explanation')
        print('        -L, --lime-feats           Instruct LIME to compute an explanation of this size')
        print('                                   Available values: [1, INT_MAX], all (default = 5)')
        print('        -m, --map-file=<string>    Path to a file containing a mapping to original feature values. (default: none)')
        print('        -M, --minimum              Compute a smallest size explanation (instead of a subset-minimal one)')
        print('        -n, --nbestims=<int>       Number of trees per class')
        print('                                   Available values: [1, INT_MAX] (default = 100)')
        print('        -N, --xnum=<int>           Number of explanations to compute')
        print('                                   Available values: [1, INT_MAX], all (default = 1)')
        print('        -o, --output=<string>      Directory where output files will be stored (default: \'temp\')')
        print('        -p,                        Preprocess categorical data')
        print('        --pfiles                   Filenames to use when preprocessing')
        print('        -q, --use-anchor           Use Anchor to compute an explanation')
        print('        -r, --rounds=<int>         Number of training rounds')
        print('                                   Available values: [1, INT_MAX] (default = 10)')
        print('        -R, --reduce=<string>      Extract an MUS from each unsatisfiable core')
        print('                                   Available values: lin, none, qxp (default = none)')
        print('        --relax=<int>              Relax the model by reducing number of weight decimal points')
        print('                                   Available values: [0, INT_MAX] (default = 0)')
        print('        --seed=<int>               Seed for random splitting')
        print('                                   Available values: [1, INT_MAX] (default = 7)')
        print('        --sep=<string>             Field separator used in input file (default = \',\')')
        print('        -s, --solver=<string>      An SMT reasoner to use')
        print('                                   Available values (smt): cvc4, mathsat, yices, z3 (default = z3)')
        print('                                   Available values (sat): g3, g4, m22, mgh, all-others-from-pysat (default = m22)')
        print('        -t, --train                Train a model of a given dataset')
        print('        -T, --trim=<int>           Trim unsatisfiable cores at most this number of times when running RC2')
        print('                                   Available values: [0, INT_MAX] (default = 0)')
        print('        --testsplit=<float>        Training and test sets split')
        print('                                   Available values: [0.0, 1.0] (default = 0.2)')
        print('        -u, --unit-mcs             Detect and block unit-size MCSes')
        print('        -v, --verbose              Increase verbosity level')
        print('        -V, --validate             Validate explanation (show that it is too optimistic)')
        print('        -w, --use-shap             Use SHAP to compute an explanation')
        print('        -x, --explain=<string>     Explain a decision for a given comma-separated sample (default: none)')
        print('        -X, --xtype=<string>       Type of explanation to compute: abductive or contrastive')
        print('                                   Available values: abd, con (default = abd)')
        print('        -z, --minz                 Apply heuristic core minimization when running RC2')
