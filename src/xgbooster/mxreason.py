#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## mxreason.py
##
##  Created on: Jun 17, 2021
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

# imported modules:
#==============================================================================
from __future__ import print_function
import collections
import copy
import decimal
from functools import reduce
from .erc2 import ERC2
import math
from pysat.examples.rc2 import RC2Stratified
from pysat.formula import CNF, WCNF, IDPool
import subprocess
import sys
import tempfile


# a named tuple for class encodings
#==============================================================================
ClassEnc = collections.namedtuple('ClassEnc', ['formula', 'leaves', 'trees'])


#
#==============================================================================
class MXReasoner:
    """
        MaxSAT-based explanation oracle. It can be called to decide whether a
        given set of feature values forbids any potential misclassifications,
        or there is a counterexample showing that the set of feature values is
        not an explanation for the prediction.
    """

    def __init__(self, encoding, target, solver='g3', oracle='int',
            am1=False, exhaust=False, minz=False, trim=0):
        """
            Magic initialiser.
        """

        self.oracles = {}   # MaxSAT solvers
        self.target = None  # index of the target class
        self.reason = None  # reason / unsatisfiable core
        self.values = collections.defaultdict(lambda: [])  # values for all the classes
        self.scores = {}  # class scores
        self.formulas = {}
        self.ortype = oracle

        # MaxSAT-oracle options
        self.am1 = am1
        self.exhaust = exhaust
        self.minz = minz
        self.trim = trim
        self.solver = solver  # keeping for alien solvers

        # doing actual initialisation
        self.init(encoding, target, solver)

    def __del__(self):
        """
            Magic destructor.
        """

        self.delete()

    def __enter__(self):
        """
            'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """

        self.delete()

    def init(self, encoding, target, solver):
        """
            Actual constructor.
        """

        # saving the target
        self.target = target

        # copying class values
        for clid in encoding:
            for lit, wght in encoding[clid].leaves:
                self.values[clid].append(tuple([lit, wght]))

        # creating the formulas and oracles
        for clid in encoding:
            if clid == self.target:
                continue

            # adding hard clauses
            self.formulas[clid] = WCNF()
            for cl in encoding[clid].formula:
                self.formulas[clid].append(cl)

            if len(encoding) > 2:
                for cl in encoding[self.target].formula:
                    self.formulas[clid].append(cl)

            # adding soft clauses and recording all the leaf values
            self.init_soft(encoding, clid)

            if self.ortype == 'int':
                # a new MaxSAT solver
                self.oracles[clid] = ERC2(self.formulas[clid], solver=solver,
                        adapt=self.am1, blo='cluster', exhaust=self.exhaust,
                        minz=self.minz, verbose=0)

    def init_soft(self, encoding, clid):
        """
            Processing the leaves and creating the set of soft clauses.
        """

        # new vpool for the leaves, and total cost
        vpool = IDPool(start_from=self.formulas[clid].nv + 1)

        # all leaves to be used in the formula, am1 constraints and cost
        wghts, atmosts, cost = collections.defaultdict(lambda: 0), [], 0

        for label in (clid, self.target):
            if label != self.target:
                coeff = 1
            else:  # this is the target class
                if len(encoding) > 2:
                    coeff = -1
                else:
                    # we don't encoding the target class if there are
                    # only two classes - it duplicates the other class
                    continue

            # here we are going to automatically detect am1 constraints
            for tree in encoding[label].trees:
                am1 = []
                for i in range(tree[0], tree[1]):
                    lit, wght = encoding[label].leaves[i]

                    # all leaves of each tree comprise an AtMost1 constraint
                    am1.append(lit)

                    # updating literal's final weight
                    wghts[lit] += coeff * wght

                atmosts.append(am1)

        # filtering out those with zero-weights
        wghts = dict(filter(lambda p: p[1] != 0, wghts.items()))

        # processing the opposite literals, if any
        i, lits = 0, sorted(wghts.keys(), key=lambda l: 2 * abs(l) + (0 if l > 0 else 1))
        while i < len(lits) - 1:
            if lits[i] == -lits[i + 1]:
                l1, l2 = lits[i], lits[i + 1]
                minw = min(wghts[l1], wghts[l2], key=lambda w: abs(w))

                # updating the weights
                wghts[l1] -= minw
                wghts[l2] -= minw

                # updating the cost if there is a conflict between l and -l
                if wghts[l1] * wghts[l2] > 0:
                    cost += abs(minw)

                i += 2
            else:
                i += 1

        # flipping literals with negative weights
        lits = list(wghts.keys())
        for l in lits:
            if wghts[l] < 0:
                cost += -wghts[l]
                wghts[-l] = -wghts[l]
                del wghts[l]

        # maximum value of the objective function
        self.formulas[clid].vmax = sum(wghts.values())

        # processing all AtMost1 constraints
        atmosts = set([tuple([l for l in am1 if l in wghts and wghts[l] != 0]) for am1 in atmosts])
        for am1 in sorted(atmosts, key=lambda am1: len(am1), reverse=True):
            if len(am1) < 2:
                continue
            cost += self.process_am1(self.formulas[clid], am1, wghts, vpool)

        # here is the start cost
        self.formulas[clid].cost = cost

        # adding remaining leaves with non-zero weights as soft clauses
        for lit, wght in wghts.items():
            if wght != 0:
                self.formulas[clid].append([ lit], weight=wght)

    def process_am1(self, formula, am1, wghts, vpool):
        """
            Detect AM1 constraints between the leaves of one tree and add the
            corresponding soft clauses to the formula.
        """

        cost = 0

        # filtering out zero-weight literals
        am1 = [l for l in am1 if wghts[l] != 0]

        # processing the literals until there is only one literal left
        while len(am1) > 1:
            minw = min(map(lambda l: wghts[l], am1))
            cost += minw * (len(am1) - 1)

            lset = frozenset(am1)
            if lset not in vpool.obj2id:
                selv = vpool.id(lset)

                # adding a new hard clause
                formula.append(am1 + [-selv])
            else:
                selv = vpool.id(lset)

            # adding a new soft clause
            formula.append([selv], weight=minw)

            # filtering out non-zero weight literals
            i = 0
            while i < len(am1):
                wghts[am1[i]] -= minw

                if wghts[am1[i]] == 0:
                    am1[i] = am1[len(am1) - 1]
                    am1.pop()
                else:
                    i += 1

        return cost

    def delete(self):
        """
            Actual destructor.
        """

        if self.oracles:
            for oracle in self.oracles.values():
                if oracle:
                    oracle.delete()

            self.oracles = {}
            self.target = None
            self.values = None
            self.reason = None
            self.scores = {}
            self.formulas = {}

    def get_coex(self, feats, full_instance=False, early_stop=False):
        """

            A call to the oracle to obtain a counterexample to a given set of
            feature values (may be a complete instance or a subset of its
            feature values). If such a counterexample exists, it is returned.
            Otherwise, the method returns None.

            Note that if None is returned, the given set of feature values is
            an abductive explanation for the prediction (not necessarily a
            minimal one).
        """

        # resetting the scores
        self.scores = {clid: 0 for clid in self.oracles}

        # updating the reason
        self.reason = set()

        if self.ortype == 'int':
            # using internal MaxSAT solver incrementally
            for clid in self.oracles:
                if clid == self.target:
                    continue

                model = self.oracles[clid].compute(feats, full_instance, early_stop)
                assert model or (early_stop and self.oracles[clid].cost > self.oracles[clid].slack), \
                        'Something is wrong, there is no MaxSAT model'

                # if misclassification, return the model
                # note that this model is not guaranteed
                # to represent the predicted class!
                if model and self.get_winner(model, clid) != self.target:
                    return model

                # otherwise, proceed to another clid
                self.reason = self.reason.union(set(self.oracles[clid].get_reason()))

            if not self.reason:
                self.reason = None

            # if no counterexample exists, return None
        else:
            # here we start an external MaxSAT solver every time
            for clid in self.formulas:
                if clid == self.target:
                    continue

                if self.ortype == 'ext':  # external RC2
                    with RC2Stratified(self.formulas[clid], solver='g3',
                            adapt=self.am1, blo='div', exhaust=self.exhaust,
                            incr=False, minz=self.minz, nohard=False,
                            trim=self.trim, verbose=0) as rc2:

                        # adding more hard clauses on top
                        for lit in feats:
                            rc2.add_clause([lit])

                        model = rc2.compute()
                else:  # expecting 'alien' here
                    # dumping the formula into a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wcnf') as fp:
                        sz = len(self.formulas[clid].hard)
                        self.formulas[clid].hard += [[l] for l in feats]
                        self.formulas[clid].to_file(fp.name)
                        self.formulas[clid].hard = self.formulas[clid].hard[:sz]
                        fp.flush()

                        outp = subprocess.check_output(self.solver.split() + [fp.name], shell=False)
                        outp = outp.decode(encoding='ascii').split('\n')

                    # going backwards in the log and extracting the model
                    for line in range(len(outp) - 1, -1, -1):
                        line = outp[line]
                        if line.startswith('v '):
                            model = [int(l) for l in line[2:].split()]

                assert model, 'Something is wrong, there is no MaxSAT model'

                # if misclassification, return the model
                # note that this model is not guaranteed
                # to represent the predicted class!
                if self.get_winner(model, clid) != self.target:
                    return model

            # otherwise, proceed to another clid
            self.reason = set(feats)

    def get_winner(self, model, clid):
        """
            Check the values for each class and extract the prediction.
        """

        for label in (self.target, clid):
            # computing the value for the current class label
            self.scores[label] = 0

            for lit, wght in self.values[label]:
                if model[abs(lit) - 1] > 0:
                    self.scores[label] += wght

        if self.scores[clid] >= self.scores[self.target]:
            return clid

        return self.target

    def get_scores(self):
        """
            Get all the actual scores for the classes computed with the previous call.
        """

        # this makes sense only for complete instances
        assert all([score != 0 for score in self.scores.values()])

        return [self.scores[clid] for clid in range(len(self.scores))]

    def get_reason(self, v2fmap=None):
        """
            Reports the last reason (analogous to unsatisfiable core in SAT).
            If the extra parameter is present, it acts as a mapping from
            variables to original categorical features, to be used a the
            reason.
        """

        assert self.reason, 'There no reason to return!'

        if v2fmap:
            return sorted(set(v2fmap[v] for v in self.reason))
        else:
            return self.reason
