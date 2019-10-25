#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## validate.py
##
##  Created on: Jan 4, 2019
##      Author: Alexey Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import getopt
import numpy as np
import os
from pysat.formula import IDPool
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol
from pysmt.shortcuts import Equals, GE, GT, LE, LT, Real, REAL
import resource
from six.moves import range
import sys


#
#==============================================================================
class SMTValidator(object):
    """
        Validating Anchor's explanations using SMT solving.
    """

    def __init__(self, formula, feats, nof_classes, xgb):
        """
            Constructor.
        """

        self.ftids = {f: i for i, f in enumerate(feats)}
        self.nofcl = nof_classes
        self.idmgr = IDPool()
        self.optns = xgb.options

        # xgbooster will also be needed
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = Solver(name=self.xgb.options.solver)

        self.inps = []  # input (feature value) variables
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' not in f:
                self.inps.append(Symbol(f, typename=REAL))
            else:
                self.inps.append(Symbol(f, typename=BOOL))

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append(Symbol('class{0}_score'.format(c), typename=REAL))

        # theory
        self.oracle.add_assertion(formula)

        # current selector
        self.selv = None

    def prepare(self, sample, expl):
        """
            Prepare the oracle for validating an explanation given a sample.
        """

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for a new sample
        sname = ','.join([str(v).strip() for v in sample])

        # the samples should not repeat; otherwise, they will be
        # inconsistent with the previously introduced selectors
        assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)

        # adding relaxed hypotheses to the oracle
        for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
            if '_' not in inp.symbol_name():
                hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
            else:
                hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

            self.oracle.add_assertion(hypo)

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        true_output = maxoval[1]

        # forcing a misclassification, i.e. a wrong observation
        disj = []
        for i in range(len(self.outs)):
            if i != true_output:
                disj.append(GT(self.outs[i], self.outs[true_output]))
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))

        # removing all hypotheses except for those in the explanation
        hypos = []
        for i, hypo in enumerate(self.rhypos):
            j = self.ftids[self.xgb.transform_inverse_by_index(i)[0]]
            if j in expl:
                hypos.append(hypo)
        self.rhypos = hypos

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in v:
                    preamble.append('{0} = {1}'.format(f, v))
                else:
                    preamble.append(v)

            print('  explanation for:  "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.xgb.target_name[true_output]))

    def validate(self, sample, expl):
        """
            Make an effort to show that the explanation is too optimistic.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample, expl)

        # if satisfiable, then there is a counterexample
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
            inpvals = [float(model.get_py_value(i)) for i in self.inps]
            outvals = [float(model.get_py_value(o)) for o in self.outs]
            maxoval = max(zip(outvals, range(len(outvals))))

            inpvals = self.xgb.transform_inverse(np.array(inpvals))[0]
            self.coex = tuple([inpvals, maxoval[1]])
            inpvals = self.xgb.readable_sample(inpvals)

            if self.verbose:
                preamble = []
                for f, v in zip(self.xgb.feature_names, inpvals):
                    if f not in v:
                        preamble.append('{0} = {1}'.format(f, v))
                    else:
                        preamble.append(v)

                print('  explanation is incorrect')
                print('  counterexample: "IF {0} THEN {1}"'.format(' AND '.join(preamble), self.xgb.target_name[maxoval[1]]))
        else:
            self.coex = None

            if self.verbose:
                print('  explanation is correct')

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.verbose:
            print('  time: {0:.2f}'.format(self.time))

        return self.coex
