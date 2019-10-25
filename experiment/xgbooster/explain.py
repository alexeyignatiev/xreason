#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##
##  Created on: Dec 14, 2018
##      Author: Alexey Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import numpy as np
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol
from pysmt.shortcuts import Equals, GT, Int, Real, REAL
import resource
from six.moves import range
import sys


#
#==============================================================================
class SMTExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = Solver(name=options.solver)

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

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
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

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # forcing a misclassification, i.e. a wrong observation
        disj = []
        for i in range(len(self.outs)):
            if i != self.out_id:
                disj.append(GT(self.outs[i], self.outs[self.out_id]))
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in v:
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        # saving external explanation to be minimized further
        if expl_ext == None or prefer_ext:
            self.to_consider = [True for h in self.rhypos]
        else:
            eexpl = set(expl_ext)
            self.to_consider = [True if i in eexpl else False for i, h in enumerate(self.rhypos)]

        # if satisfiable, then the observation is not implied by the hypotheses
        if self.oracle.solve([self.selv] + [h for h, c in zip(self.rhypos, self.to_consider) if c]):
            print('  no implication!')
            print(self.oracle.get_model())
            sys.exit(1)

        if not smallest:
            self.compute_minimal(prefer_ext=prefer_ext)
        else:
            self.compute_smallest()

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        expl = sorted([self.sel2fid[h] for h in self.rhypos])

        if self.verbose:
            self.preamble = [self.preamble[i] for i in expl]
            print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.xgb.target_name[self.out_id]))
            print('  # hypos left:', len(self.rhypos))
            print('  time: {0:.2f}'.format(self.time))

        return expl

    def compute_minimal(self, prefer_ext=False):
        """
            Compute any subset-minimal explanation.
        """

        i = 0

        if not prefer_ext:
            # here, we want to reduce external explanation

            # filtering out unnecessary features if external explanation is given
            self.rhypos = [h for h, c in zip(self.rhypos, self.to_consider) if c]
        else:
            # here, we want to compute an explanation that is preferred
            # to be similar to the given external one
            # for that, we try to postpone removing features that are
            # in the external explanation provided

            rhypos  = [h for h, c in zip(self.rhypos, self.to_consider) if not c]
            rhypos += [h for h, c in zip(self.rhypos, self.to_consider) if c]
            self.rhypos = rhypos

        # simple deletion-based linear search
        while i < len(self.rhypos):
            to_test = self.rhypos[:i] + self.rhypos[(i + 1):]

            if self.oracle.solve([self.selv] + to_test):
                i += 1
            else:
                self.rhypos = to_test

    def compute_smallest(self):
        """
            Compute a cardinality-minimal explanation.
        """

        # result
        rhypos = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]]) as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    hitman.hit([i])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.rhypos))).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        i = self.sel2fid[self.rhypos[h]]
                        if '_' not in self.inps[i].symbol_name():
                            # feature variable and its expected value
                            var, exp = self.inps[i], self.sample[i]

                            # true value
                            true_val = float(model.get_py_value(var))

                            if not exp - 0.001 <= true_val <= exp + 0.001:
                                unsatisfied.append(h)
                            else:
                                hset.append(h)
                        else:
                            for vid in self.sel2vid[self.rhypos[h]]:
                                var, exp = self.inps[vid], int(self.sample[vid])

                                # true value
                                true_val = int(model.get_py_value(var))

                                if exp != true_val:
                                    unsatisfied.append(h)
                                    break
                            else:
                                hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset] + [self.rhypos[h]]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    self.rhypos = [self.rhypos[i] for i in hset]
                    break
