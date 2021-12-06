#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## encode.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
import collections
from decimal import Decimal
from .mxreason import MXReasoner, ClassEnc
import numpy as np
from pysat.card import *
from pysat.formula import IDPool, CNF
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
        self.intvs, self.imaps, self.ivars, self.lvars = None, None, None, None

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
            value = Real(tree.values) if not self.optns.relax else Real(round(tree.values, self.optns.relax))
            if prefix:
                self.enc.append(Implies(And(prefix), Equals(tvar, value)))
            else:
                self.enc.append(Equals(tvar, value))

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
        self.intvs = {'{0}'.format(i): set([]) for i in self.xgb.extended_feature_names_as_array_strings}

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
        if self.optns.relax:
            cscores = [round(v, self.optns.relax) for v in cscores]

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


#
#==============================================================================
class MXEncoder(SMTEncoder):
    """
        Encoder for the MaxSAT-based reasoner.
    """

    def __init__(self, model, feats, nof_classes, xgb, from_file=None):
        """
            Initialiser.
        """

        super(MXEncoder, self).__init__(model, feats, nof_classes, xgb, None)

        # variable to feature id
        self.vid2fid = {}

        if from_file:
            self.load_from(from_file)

    def traverse(self, tree, clid, prefix=[]):
        """
            Traverse a tree and encode each node.
        """

        if tree.children:
            var = self.encode_node(tree)

            self.traverse(tree.children[0], clid, prefix + [ var])
            self.traverse(tree.children[1], clid, prefix + [-var])
        else:  # leaf node
            leaf = self.idmgr.id(tuple(sorted(prefix)))

            if prefix:
                # encoding the path only if necessary
                if leaf not in self.enc['paths']:
                    for v in prefix:
                        self.enc['paths'][leaf].append([v, -leaf])
                    self.enc['paths'][leaf].append([-v for v in prefix] + [leaf])

                # copying its encoding into the current class encoding
                self.enc[clid].formula.extend(self.enc['paths'][leaf])
            else:
                # we may need to consider this hard clause!
                self.enc[clid].formula.append([leaf])

            # adding the leaf with its weight
            value = Decimal(str(tree.values)) if not self.optns.relax else round(Decimal(str(tree.values)), self.optns.relax)
            self.enc[clid].leaves.append((leaf, value))

    def encode_node(self, node):
        """
            Encode a node of a tree.
        """

        feat, fval = node.name, node.threshold
        intv = self.imaps[feat][fval]

        return self.lvars[feat][intv]

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
        self.intvs = {'{0}'.format(i): set([]) for i in self.xgb.extended_feature_names_as_array_strings}

        for tree in self.ensemble.trees:
            traverse_intervals(tree)

        # filtering out variables that do not appear in the trees
        self.intvs = dict(filter(lambda x: len(x[1]) != 0, self.intvs.items()))

        # sorting the intervals
        self.intvs = {f: sorted(self.intvs[f]) + ['+'] for f in six.iterkeys(self.intvs)}

        self.imaps, self.ivars, self.lvars = {}, {}, {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {intvs[i]: i for i in range(len(intvs))}
            self.ivars[feat] = [None for i in range(len(intvs))]
            self.lvars[feat] = [None for i in range(len(intvs))]

            # separate case of the first interval
            self.lvars[feat][0] = self.idmgr.id('{0}_lvar{1}'.format(feat, 0))
            self.ivars[feat][0] = self.lvars[feat][0]

            # main part: order + domain encoding
            for i in range(1, len(intvs) - 1):
                lvar = self.idmgr.id('{0}_lvar{1}'.format(feat, i))
                ivar = self.idmgr.id('{0}_ivar{1}'.format(feat, i))
                prev = self.lvars[feat][i - 1]

                # order encoding
                self.enc['common'].append([-prev, lvar])

                # domain encoding
                self.enc['common'].append([-ivar, -prev])
                self.enc['common'].append([-ivar,  lvar])
                self.enc['common'].append([ ivar,  prev, -lvar])

                # saving the variables
                self.lvars[feat][i] = lvar
                self.ivars[feat][i] = ivar

            # separate case of the last interval (till "+inf")
            self.lvars[feat][-1] = -self.lvars[feat][-2]
            self.ivars[feat][-1] =  self.lvars[feat][-1]

            # finally, enforcing that there is at least one interval
            if len(intvs) > 2:
                self.enc['common'].append(self.ivars[feat])

        # mapping variable ids to feature ids
        for feat in self.ivars:
            for v, ub in zip(self.ivars[feat], self.intvs[feat]):
                self.vid2fid[v] = (feat, ub)

    def encode(self):
        """
            Do the job.
        """

        # creatin initially empty class encodings
        self.enc = {}
        for j in range(self.nofcl):
            self.enc[j] = ClassEnc(formula=CNF(), leaves=[], trees=[])

        # common clauses shared by all classes
        self.enc['common'] = []

        # path encodings
        self.enc['paths'] = collections.defaultdict(lambda: [])

        # getting a tree ensemble
        self.ensemble = TreeEnsemble(self.model,
                self.xgb.extended_feature_names_as_array_strings,
                nb_classes=self.nofcl)

        # we have to consider interval-based encoding, traverse all
        # trees and extract all possible intervals for each feature
        self.compute_intervals()

        # traversing and encoding each tree
        for i, tree in enumerate(self.ensemble.trees):
            # getting class id
            clid = i % self.nofcl

            # determining the beginning of the newly created leaves
            beg = len(self.enc[clid].leaves)

            # encoding the tree
            self.traverse(tree, clid, prefix=[])

            # determining the end of the newly created leaves
            end = len(self.enc[clid].leaves)

            # leaf vars for the current tree
            leaves = list(map(lambda t: t[0], self.enc[clid].leaves[beg:end]))

            # recording ids of the leaves for each tree
            self.enc[clid].trees.append((beg, end))

            # adding an EqualsOne constraint for the leaves of each tree
            self.enc[clid].formula.extend(CardEnc.equals(leaves,
                    vpool=self.idmgr, encoding=self.optns.cardenc))

        # creating variable positions (in the consequtive list of features)
        self.make_varpos()

        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        categories = collections.defaultdict(lambda: [])
        expected = collections.defaultdict(lambda: 0)
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' in f:
                if f in self.ivars:
                    categories[f.split('_')[0]].append(self.ivars[f][1])
                expected[f.split('_')[0]] += 1
        for c, feats in six.iteritems(categories):
            if len(feats) > 1:
                if len(feats) == expected[c]:
                    self.enc['common'].extend(CardEnc.equals(feats,
                        vpool=self.idmgr, encoding=self.optns.cardenc))
                else:
                    self.enc['common'].extend(CardEnc.atmost(feats,
                        vpool=self.idmgr, encoding=self.optns.cardenc))

        # adding all common clauses to all the class formulas
        for j in range(self.nofcl):
            self.enc[j].formula.extend(self.enc['common'])
            self.enc[j].formula.nv = self.idmgr.top

        # number of assertions
        nof_clauses = sum([len(self.enc[j].formula.clauses) for j in range(self.nofcl)])

        # number of variables
        nof_vars = self.idmgr.top

        if self.optns.verb:
            print('encoding vars:', nof_vars)
            print('encoding clauses: {0} ({1} path + {2} common)'.format(nof_clauses, nof_clauses - len(self.enc['common']), len(self.enc['common'])))

        # deleting unnecessary clauses
        del self.enc['common']
        del self.enc['paths']

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
        if self.optns.relax:
            cscores = [round(v, self.optns.relax) for v in cscores]
        zscores = [(i, s) for i, s in enumerate(cscores)]
        cwinner = max(zscores, key=lambda pair: pair[1])[0]

        # second, get the scores computed with the use of the encoding
        hypos = self.get_literals(sample)

        # now, getting the model
        if self.optns.encode == 'mxa':
            ortype = 'alien'
        elif self.optns.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'
        with MXReasoner(self.enc, cwinner, solver=self.optns.solver,
                oracle=ortype) as x:
            assert x.get_coex(hypos) == None, 'Wrong class predicted by the encoding'
            escores = x.get_scores()

        assert all(map(lambda c, e: abs(Decimal(c) - e) <= Decimal(0.001), cscores, escores)), \
                'wrong prediction: {0} vs {1}'.format(cscores, escores)

        if self.optns.verb:
            print('xgb scores:', cscores)
            print('enc scores:', [float(str(e)) for e in escores])

    def make_varpos(self):
        """
            Traverse all the vars and get their positions in the list of inputs.
        """

        self.vpos, pos = {}, 0

        for feat in self.ivars:
            if '_' in feat or len(self.ivars[feat]) == 2:
                for lit in self.ivars[feat]:
                    if abs(lit) not in self.vpos:
                        self.vpos[abs(lit)] = pos
                        pos += 1
            else:
                for lit in self.ivars[feat]:
                    if abs(lit) not in self.vpos:
                        self.vpos[abs(lit)] = pos
                pos += 1

    def get_literals(self, sample):
        """
            Translate an instance to a list of propositional literals.
        """

        lits = []

        sample_internal = list(self.xgb.transform(sample)[0])
        for feat, fval in zip(self.xgb.extended_feature_names_as_array_strings, sample_internal):
            if feat in self.intvs:
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[feat], self.ivars[feat]):
                    if ub == '+' or fval < ub:
                        lits.append(fvar)
                        break

        return lits

    def get_instance(self, lits):
        """
            Translate a list of literals over input variables to an instance.
        """

        return None

    def save_to(self, outfile):
        """
            Save the encoding into a file with a given name.
        """

        if outfile.endswith('.txt'):
            outfile = outfile[:-3] + 'cnf'

        formula = CNF()

        # comments
        formula.comments = ['c features: {0}'.format(', '.join(self.feats)),
                'c classes: {0}'.format(self.nofcl)]

        for clid in self.enc:
            formula.comments += ['c clid starts: {0} {1}'.format(clid, len(formula.clauses))]
            for leaf in self.enc[clid].leaves:
                formula.comments += ['c leaf: {0} {1} {2}'.format(clid, *leaf)]
            formula.clauses.extend(self.enc[clid].formula.clauses)

        for f in self.xgb.extended_feature_names_as_array_strings:
            if f in self.intvs:
                c = 'c i {0}: '.format(f)
                c += ', '.join(['"{0}" <-> "{1}"'.format(u, v) for u, v in zip(self.intvs[f], self.ivars[f])])
            else:
                c = 'c i {0}: none'.format(f)

            formula.comments.append(c)

        formula.to_file(outfile)

    def load_from(self, infile):
        """
            Loads the encoding from an input file.
        """

        self.enc = CNF(from_file=infile)

        # empty intervals for the standard encoding
        self.intvs, self.imaps, self.ivars = {}, {}, {}

        for line in self.enc.comments:
            if line.startswith('c i ') and 'none' not in line:
                f, arr = line[4:].strip().split(': ', 1)
                f = f.replace('-', '_')
                self.intvs[f], self.imaps[f], self.ivars[f] = [], {}, []

                for i, pair in enumerate(arr.split(', ')):
                    ub, symb = pair.split(' <-> ')
                    ub = ub.strip('"')
                    symb = symb.strip('"')

                    if ub[0] != '+':
                        ub = float(ub)

                    self.intvs[f].append(ub)
                    self.ivars[f].append(symb)
                    self.imaps[f][ub] = i

            elif line.startswith('c features:'):
                self.feats = line[11:].strip().split(', ')
            elif line.startswith('c classes:'):
                self.nofcl = int(line[10:].strip())
