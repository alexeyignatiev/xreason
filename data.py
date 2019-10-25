#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## data.py
##
##  Created on: Sep 20, 2017
##      Author: Alexey Ignatiev, Nina Narodytska
##      E-mail: aignatiev@ciencias.ulisboa.pt, narodytska@vmware.com
##

#
#==============================================================================
from __future__ import print_function
import collections
import itertools
import os, pickle
import six
from six.moves import range
import numpy as np


#
#==============================================================================
class Data(object):
    """
        Class for representing data (transactions).
    """

    def __init__(self, filename=None, fpointer=None, mapfile=None,
            separator=' ', use_categorical = False):
        """
            Constructor and parser.
        """

        self.names = None
        self.nm2id = None
        self.samps = None
        self.wghts = None
        self.feats = None
        self.fvmap = None
        self.ovmap = {}
        self.fvars = None
        self.fname = filename
        self.mname = mapfile
        self.deleted = set([])

        if filename:
            with open(filename, 'r') as fp:
                self.parse(fp, separator)
        elif fpointer:
            self.parse(fpointer, separator)

        if self.mname:
            self.read_orig_values()

        # check if we have extra info about categorical_features

        if (use_categorical):
            extra_file = filename+".pkl"
            try:
                f =  open(extra_file, "rb")
                print("Attempt: loading extra data from ", extra_file)
                extra_info = pickle.load(f)
                print("loaded")
                f.close()
                self.categorical_features = extra_info["categorical_features"]
                self.categorical_names = extra_info["categorical_names"]
                self.class_names = extra_info["class_names"]
                self.categorical_onehot_names  = extra_info["categorical_names"].copy()

                for i, name in enumerate(self.class_names):
                    self.class_names[i] = str(name).replace("b'","'")
                for c in self.categorical_names.items():
                    clean_feature_names = []
                    for i, name in enumerate(c[1]):
                        name = str(name).replace("b'","'")
                        clean_feature_names.append(name)
                    self.categorical_names[c[0]] = clean_feature_names

            except Exception as e:
                f.close()
                print("Please provide info about categorical features or omit option -c", e)
                exit()

    def parse(self, fp, separator):
        """
            Parse input file.
        """

        # reading data set from file
        lines = fp.readlines()

        # reading preamble
        self.names = lines[0].strip().split(separator)
        self.feats = [set([]) for n in self.names]
        del(lines[0])

        # filling name to id mapping
        self.nm2id = {name: i for i, name in enumerate(self.names)}

        self.nonbin2bin = {}
        for name in self.nm2id:
            spl = name.rsplit(':',1)
            if (spl[0] not in self.nonbin2bin):
                self.nonbin2bin[spl[0]] = [name]
            else:
                self.nonbin2bin[spl[0]].append(name)

        # reading training samples
        self.samps, self.wghts = [], []

        for line, w in six.iteritems(collections.Counter(lines)):
            sample = line.strip().split(separator)
            for i, f in enumerate(sample):
                if f:
                    self.feats[i].add(f)
            self.samps.append(sample)
            self.wghts.append(w)

        # direct and opposite mappings for items
        idpool = itertools.count(start=0)
        FVMap = collections.namedtuple('FVMap', ['dir', 'opp'])
        self.fvmap = FVMap(dir={}, opp={})

        # mapping features to ids
        for i in range(len(self.names) - 1):
            feats = sorted(list(self.feats[i]), reverse=True)
            if len(feats) > 2:
                for l in feats:
                    self.fvmap.dir[(self.names[i], l)] = l
            else:
                self.fvmap.dir[(self.names[i], feats[0])] = 1
                if len(feats) == 2:
                    self.fvmap.dir[(self.names[i], feats[1])] = 0

        # opposite mapping
        for key, val in six.iteritems(self.fvmap.dir):
            self.fvmap.opp[val] = key

        # determining feature variables (excluding class variables)
        for v, pair in six.iteritems(self.fvmap.opp):
            if pair[0] == self.names[-1]:
                self.fvars = v - 1
                break

    def read_orig_values(self):
        """
            Read original values for all the features.
            (from a separate CSV file)
        """

        self.ovmap = {}

        for line in open(self.mname, 'r'):
            featval, bits = line.strip().split(',')
            feat, val = featval.split(':')

            for i, b in enumerate(bits):
                f = '{0}:b{1}'.format(feat, i + 1)
                v = self.fvmap.dir[(f, '1')]

                if v not in self.ovmap:
                    self.ovmap[v] = [feat]

                if -v not in self.ovmap:
                    self.ovmap[-v] = [feat]

                self.ovmap[v if b == '1' else -v].append(val)
