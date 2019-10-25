#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## encode.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import collections
from pysat.formula import IDPool
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import And, BOOL, Iff, Implies, Not, Or, Symbol, get_model
from pysmt.shortcuts import Equals, ExactlyOne, LT, Plus, REAL, Real, write_smtlib
from .tree import TreeEnsemble, scores_tree
import six
from six.moves import range

try:  # for Python2
    from cStringIO import StringIO
except ImportError:  # for Python3
    from io import StringIO


#
#==============================================================================
class SMTEncoder(object):
    """
        Encoder of XGBoost tree ensembles into SMT.
    """

    def __init__(self, model, feats, nof_classes, xgb, from_file=None):
        """
            Constructor.
        """

        self.model = model
        self.feats = {f: i for i, f in enumerate(feats)}
        self.nofcl = nof_classes
        self.idmgr = IDPool()
        self.optns = xgb.options

        # xgbooster will also be needed
        self.xgb = xgb

        # for interval-based encoding
        self.intvs, self.imaps, self.ivars = None, None, None

        if from_file:
            self.load_from(from_file)

    def traverse(self, tree, tvar, prefix=[]):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            pos, neg = self.encode_node(tree)

            self.traverse(tree.children[0], tvar, prefix + [pos])
            self.traverse(tree.children[1], tvar, prefix + [neg])
        else:  # leaf node
            if prefix:
                self.enc.append(Implies(And(prefix), Equals(tvar, Real(tree.values))))
            else:
                self.enc.append(Equals(tvar, Real(tree.values)))

    def encode_node(self, node):
        """
            Encode a node of a tree.
        """

        if '_' not in node.name:
            # continuous features => expecting an upper bound
            # feature and its upper bound (value)
            f, v = node.name, node.threshold

            existing = True if tuple([f, v]) in self.idmgr.obj2id else False
            vid = self.idmgr.id(tuple([f, v]))
            bv = Symbol('bvar{0}'.format(vid), typename=BOOL)

            if not existing:
                if self.intvs:
                    d = self.imaps[f][v] + 1
                    pos, neg = self.ivars[f][:d], self.ivars[f][d:]
                    self.enc.append(Iff(bv, Or(pos)))
                    self.enc.append(Iff(Not(bv), Or(neg)))
                else:
                    fvar, fval = Symbol(f, typename=REAL), Real(v)
                    self.enc.append(Iff(bv, LT(fvar, fval)))

            return bv, Not(bv)
        else:
            # all features are expected to be categorical and
            # encoded with one-hot encoding into Booleans
            # each node is expected to be of the form: f_i < 0.5
            bv = Symbol(node.name, typename=BOOL)

            # left branch is positive,  i.e. bv is true
            # right branch is negative, i.e. bv is false
            return Not(bv), bv

    def compute_intervals(self):
        """
            Traverse all trees in the ensemble and extract intervals for each
            feature.

            At this point, the method only works for numerical datasets!
        """

        def traverse_intervals(tree):
            """
                Auxiliary function. Recursive tree traversal.
            """

            if tree.children:
                f = tree.name
                v = tree.threshold
                self.intvs[f].add(v)

                traverse_intervals(tree.children[0])
                traverse_intervals(tree.children[1])

        # initializing the intervals
        self.intvs = {'f{0}'.format(i): set([]) for i in range(len(self.feats))}

        for tree in self.ensemble.trees:
            traverse_intervals(tree)

        # OK, we got all intervals; let's sort the values
        self.intvs = {f: sorted(self.intvs[f]) + ['+'] for f in six.iterkeys(self.intvs)}

        self.imaps, self.ivars = {}, {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {}
            self.ivars[feat] = []
            for i, ub in enumerate(intvs):
                self.imaps[feat][ub] = i

                ivar = Symbol(name='{0}_intv{1}'.format(feat, i), typename=BOOL)
                self.ivars[feat].append(ivar)

    def encode(self):
        """
            Do the job.
        """

        self.enc = []

        # getting a tree ensemble
        self.ensemble = TreeEnsemble(self.model,
                self.xgb.extended_feature_names_as_array_strings,
                nb_classes=self.nofcl)

        # introducing class score variables
        csum = []
        for j in range(self.nofcl):
            cvar = Symbol('class{0}_score'.format(j), typename=REAL)
            csum.append(tuple([cvar, []]))

        # if targeting interval-based encoding,
        # traverse all trees and extract all possible intervals
        # for each feature
        if self.optns.encode == 'smtbool':
            self.compute_intervals()

        # traversing and encoding each tree
        for i, tree in enumerate(self.ensemble.trees):
            # getting class id
            clid = i % self.nofcl

            # encoding the tree
            tvar = Symbol('tr{0}_score'.format(i + 1), typename=REAL)
            self.traverse(tree, tvar, prefix=[])

            # this tree contributes to class with clid
            csum[clid][1].append(tvar)

        # encoding the sums
        for pair in csum:
            cvar, tvars = pair
            self.enc.append(Equals(cvar, Plus(tvars)))

        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        categories = collections.defaultdict(lambda: [])
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' in f:
                categories[f.split('_')[0]].append(Symbol(name=f, typename=BOOL))
        for c, feats in six.iteritems(categories):
            self.enc.append(ExactlyOne(feats))

        # number of assertions
        nof_asserts = len(self.enc)

        # making conjunction
        self.enc = And(self.enc)

        # number of variables
        nof_vars = len(self.enc.get_free_variables())

        if self.optns.verb:
            print('encoding vars:', nof_vars)
            print('encoding asserts:', nof_asserts)

        return self.enc, self.intvs, self.imaps, self.ivars

    def test_sample(self, sample):
        """
            Check whether or not the encoding "predicts" the same class
            as the classifier given an input sample.
        """

        # first, compute the scores for all classes as would be
        # predicted by the classifier

        # score arrays computed for each class
        csum = [[] for c in range(self.nofcl)]

        if self.optns.verb:
            print('testing sample:', list(sample))

        sample_internal = list(self.xgb.transform(sample)[0])

        # traversing all trees
        for i, tree in enumerate(self.ensemble.trees):
            # getting class id
            clid = i % self.nofcl

            # a score computed by the current tree
            score = scores_tree(tree, sample_internal)

            # this tree contributes to class with clid
            csum[clid].append(score)

        # final scores for each class
        cscores = [sum(scores) for scores in csum]

        # second, get the scores computed with the use of the encoding

        # asserting the sample
        hypos = []

        if not self.intvs:
            for i, fval in enumerate(sample_internal):
                feat, vid = self.xgb.transform_inverse_by_index(i)
                fid = self.feats[feat]

                if vid == None:
                    fvar = Symbol('f{0}'.format(fid), typename=REAL)
                    hypos.append(Equals(fvar, Real(float(fval))))
                else:
                    fvar = Symbol('f{0}_{1}'.format(fid, vid), typename=BOOL)
                    if int(fval) == 1:
                        hypos.append(fvar)
                    else:
                        hypos.append(Not(fvar))
        else:
            for i, fval in enumerate(sample_internal):
                feat, _ = self.xgb.transform_inverse_by_index(i)
                feat = 'f{0}'.format(self.feats[feat])

                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[feat], self.ivars[feat]):
                    if ub == '+' or fval < ub:
                        hypos.append(fvar)
                        break
                else:
                    assert 0, 'No proper interval found for {0}'.format(feat)

        # now, getting the model
        escores = []
        model = get_model(And(self.enc, *hypos), solver_name=self.optns.solver)
        for c in range(self.nofcl):
            v = Symbol('class{0}_score'.format(c), typename=REAL)
            escores.append(float(model.get_py_value(v)))

        assert all(map(lambda c, e: abs(c - e) <= 0.001, cscores, escores)), \
                'wrong prediction: {0} vs {1}'.format(cscores, escores)

        if self.optns.verb:
            print('xgb scores:', cscores)
            print('enc scores:', escores)

    def save_to(self, outfile):
        """
            Save the encoding into a file with a given name.
        """

        if outfile.endswith('.txt'):
            outfile = outfile[:-3] + 'smt2'

        write_smtlib(self.enc, outfile)

        # appending additional information
        with open(outfile, 'r') as fp:
            contents = fp.readlines()

        # comments
        comments = ['; features: {0}\n'.format(', '.join(self.feats)),
                '; classes: {0}\n'.format(self.nofcl)]

        if self.intvs:
            for f in self.xgb.extended_feature_names_as_array_strings:
                c = '; i {0}: '.format(f)
                c += ', '.join(['{0}<->{1}'.format(u, v) for u, v in zip(self.intvs[f], self.ivars[f])])
                comments.append(c + '\n')

        contents = comments + contents
        with open(outfile, 'w') as fp:
            fp.writelines(contents)

    def load_from(self, infile):
        """
            Loads the encoding from an input file.
        """

        with open(infile, 'r') as fp:
            file_content = fp.readlines()

        # empty intervals for the standard encoding
        self.intvs, self.imaps, self.ivars = {}, {}, {}

        for line in file_content:
            if line[0] != ';':
                break
            elif line.startswith('; i '):
                f, arr = line[4:].strip().split(': ', 1)
                f = f.replace('-', '_')
                self.intvs[f], self.imaps[f], self.ivars[f] = [], {}, []

                for i, pair in enumerate(arr.split(', ')):
                    ub, symb = pair.split('<->')

                    if ub[0] != '+':
                        ub = float(ub)
                    symb = Symbol(symb, typename=BOOL)

                    self.intvs[f].append(ub)
                    self.ivars[f].append(symb)
                    self.imaps[f][ub] = i

            elif line.startswith('; features:'):
                self.feats = line[11:].strip().split(', ')
            elif line.startswith('; classes:'):
                self.nofcl = int(line[10:].strip())

        parser = SmtLibParser()
        script = parser.get_script(StringIO(''.join(file_content)))

        self.enc = script.get_last_formula()

    def access(self):
        """
            Get access to the encoding, features names, and the number of
            classes.
        """

        return self.enc, self.intvs, self.imaps, self.ivars, self.feats, self.nofcl
