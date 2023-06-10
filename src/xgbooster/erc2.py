#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## erc2.py
##
##  Created on: Jun 16, 2021
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

# imported modules:
#==============================================================================
from __future__ import print_function
import collections
from copy import deepcopy
from dataclasses import dataclass
import functools
import itertools
from math import ceil, copysign
import namedlist
from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.formula import IDPool
from pysat.solvers import Solver
import sys


# a named tuple for storing the information associated with a core
#==============================================================================
assert sys.version_info.major >= 3, 'Python 2 is not supported'
if sys.version_info.major == 3 and sys.version_info.minor < 10:
    CoreInfo = namedlist.namedlist('CoreInfo', ['tobj', 'tbnd', 'sz', 'lits', 'reasons'])
else:  # namedlist does not work with Python >= 3.10
    @dataclass
    class CoreInfo:
        """ Dataclass for storing the information associated with a core. """
        tobj: dict
        tbnd: dict
        sz: int
        lits: 'typing.Any'
        reasons: 'typing.Any'


#
#==============================================================================
class CoreOracle(Solver):
    """
        This class is for storing the dependencies between unsatisfiable cores
        detected by RC2. It can be used to determine the cores that can be
        reused given the current assumption literals.
    """

    def __init__(self, name='m22'):
        """
            Initializer.
        """

        # first, calling base class method
        super(CoreOracle, self).__init__(name=name)

        # we are going to redefine the variables so that there are no conflicts
        self.pool = IDPool(start_from=1)

        # this is a global selector; all clauses should have it
        self.selv = self.pool.id()

        # here are all the known sum literals
        self.lits = set([])

    def delete(self):
        """
            Destructor.
        """

        # first, calling base class method
        super(CoreOracle, self).delete()

        # setting the vars to None
        self.pool, self.selv, self.lits = None, None, None

    def record(self, core, slit):
        """
            Record a new fact (core -> slit). The "core" lits must be already
            negated.
        """

        # translating the literals into internal representation
        cl = [int(copysign(self.pool.id(abs(l)), l)) for l in core + [slit]]

        # adding the clause
        self.add_clause([-self.selv] + cl)

        # storing the sum for future filtering
        self.lits.add(int(copysign(self.pool.id(abs(slit)), slit)))

    def get_active(self, assumps):
        """
            Check what cores are propagated given a list of assumptions.
        """

        # translating assumptions into internal representation
        assumps = [int(copysign(self.pool.id(abs(l)), l)) for l in assumps]

        # doing the actual propagation
        st, props = self.propagate(assumptions=[self.selv] + assumps, phase_saving=2)
        assert st, 'Something is wrong. The core-deps formula is unsatisfiable'

        # processing literals and returning the result; note
        # that literals must remain in the right order here
        return tuple(map(lambda l: int(copysign(self.pool.obj(abs(l)), l)),
            filter(lambda l: l in self.lits, props[1:])))


#
#==============================================================================
class ERC2(RC2Stratified):
    """
        This is an extended version of RC2, which can disable some of the soft
        clauses and reuse unsatisfiable cores from the previous MaxSAT calls.
    """

    def __init__(self, formula, solver='g3', adapt=False, blo='div',
            exhaust=False, incr=False, minz=False, trim=0, verbose=0):
        """
            Initialiser.
        """

        super(ERC2, self).__init__(formula, solver=solver, adapt=adapt,
                blo=blo, exhaust=exhaust, incr=incr, minz=minz, nohard=True,
                trim=trim, verbose=verbose)

        # to support earlier versions of PySAT/RC2
        if not hasattr(self, 'swgt'):
            self.swgt = {}
            self.process_sums_ext = self.process_sums_old
        else:  # for PySAT version >= 0.1.8.dev7
            self.process_sums_ext = self.process_sums_new

        # here is the slack for the total cost
        self.slack = formula.vmax - formula.cost

        # cost of approximate solution
        self.ubcost = None

        # saving the state
        self.save_state()

        # extra hard assumptions and their set
        self.ehard, self.ehset = [], set()

        # previously and currently obtained cores
        self.cores = {}  # this mapping has the structure:
                         # frozenset(core) -> {
                         #     tuple 'rvar',
                         #     dict 'tobj' = {frozenset(rels): ITotalizer},
                         #     dict 'tbnd' = {frozenset(rels): int},
                         # }

        # all the known cores
        self.cores = collections.defaultdict(lambda: CoreInfo(tobj=None,
            tbnd=None, sz=0, lits=set(), reasons=[]))

        # this is a checker of possible cores to reuse
        self.cchecker = CoreOracle(name=solver)

        # here is the reason variable
        self.reason = None

        # here is a mapping from frozenset() to a totalizer object
        self.tots = {}

        # set of all possible full-instances considered so far
        self.instances = set()

    def delete(self):
        """
            Destructor.
        """

        # calling base class destructor
        super(ERC2, self).delete()

        # deleting the core-deps oracle
        if self.cchecker:
            self.cchecker.delete()

    def save_state(self):
        """
            Saving the base case state of the solver after the base case
            is finished.
        """

        self.cost_copy = self.cost
        self.sels_copy = deepcopy(self.sels)
        self.sset_copy = deepcopy(self.sels_set)
        self.smap_copy = deepcopy(self.smap)
        self.sall_copy = deepcopy(self.sall)
        self.s2cl_copy = deepcopy(self.s2cl)
        self.sneg_copy = deepcopy(self.sneg)
        self.wght_copy = deepcopy(self.wght)
        self.swgt_copy = deepcopy(self.swgt)
        self.sums_copy = deepcopy(self.sums)
        self.bnds_copy = deepcopy(self.bnds)
        self.levl_copy = self.levl  # initial optimization level
        self.wstr_copy = deepcopy(self.wstr)
        self.blop_copy = deepcopy(self.blop)  # a list of blo levels
        self.sdiv_copy = self.sdiv
        self.done_copy = self.done

        # backing up selectors
        self.bckp_copy = deepcopy(self.bckp)
        self.sbck_copy = deepcopy(self.bckp_set)

        self.slck_copy = self.slack

    def load_state(self, extra_hard):
        """
            Loading the base case state of the solver whenever necessary.
        """

        self.cost = self.cost_copy
        self.sels = deepcopy(self.sels_copy)
        self.sels_set = deepcopy(self.sset_copy)
        self.smap = deepcopy(self.smap_copy)
        self.sall = deepcopy(self.sall_copy)
        self.s2cl = deepcopy(self.s2cl_copy)
        self.sneg = deepcopy(self.sneg_copy)
        self.wght = deepcopy(self.wght_copy)
        self.swgt = deepcopy(self.swgt_copy)
        self.sums = deepcopy(self.sums_copy)
        self.bnds = deepcopy(self.bnds_copy)
        self.levl = self.levl_copy
        self.wstr = deepcopy(self.wstr_copy)
        self.blop = deepcopy(self.blop_copy)  # a list of blo levels
        self.sdiv = self.sdiv_copy
        self.done = self.done_copy
        self.bckp = deepcopy(self.bckp_copy)
        self.bckp_set = deepcopy(self.sbck_copy)

        self.slack = self.slck_copy
        self.ubcost = None

        # copying extra hard assumptions
        if extra_hard:
            self.ehard = [self._map_extlit(l) for l in extra_hard]
        else:
            self.ehard = []
        self.ehset = set(self.ehard)

        self.reason = set()

    def get_reason(self):
        """
            Return the set of extra objective clauses that participate in
            the cores.
        """

        return self.reason

    def compute(self, extra_hard=None, full_instance=False, early_stop=False):
        """
            Interface for a MaxSAT call. Here we either call the standard
            method of RC2 or a modified one, depending on whether we are
            making a "base case" call. The base case is when there are no
            extra soft clauses.
        """

        # remembering if we need to terminate early
        self.estop = early_stop

        # additional processing in case of a full-instance
        if full_instance:
            assumps = frozenset(extra_hard)
            if assumps in self.instances:
                full_instance = False
            else:
                # if a new full instance, create a new core-deps oracle
                self.cchecker.delete()
                self.cchecker = CoreOracle(name=self.solver)

                self.instances.add(assumps)

        # first, loading the solver state
        self.load_state(extra_hard)

        # first attempt to get an optimization level
        self.next_level()

        while self.levl != None and self.done < len(self.blop):
            # add more clauses
            self.done = self.activate_clauses(self.done)

            if self.verbose > 1:
                print('c wght str:', self.blop[self.levl])

            # call RC2
            if self.compute_ext(full_instance) == False:
                return

            # updating the list of distinct weight levels
            self.blop = sorted([w for w in self.wstr], reverse=True)

            if self.done < len(self.blop):
                if self.estop and not full_instance:
                    self.ubcost = self.get_cost(self.oracle.get_model())
                    if self.ubcost <= self.slack:
                        break

                if self.verbose > 1:
                    print('c curr opt:', self.cost)

                # done with this level
                if self.hard:
                    # harden the clauses if necessary
                    self.finish_level()

                self.levl += 1

                # get another level
                self.next_level()

                if self.verbose > 1:
                    print('c')

        if not self.reason:
            self.reason = None

        # extracting a model
        self.model = self.oracle.get_model()

        if self.model is None and self.topv == 0:
            # we seem to have been given an empty formula
            # so let's transform the None model returned to []
            self.model = []

        self.model = filter(lambda l: abs(l) in self.vmap.i2e, self.model)
        self.model = map(lambda l: int(copysign(self.vmap.i2e[abs(l)], l)), self.model)
        self.model = sorted(self.model, key=lambda l: abs(l))

        return self.model

    def compute_ext(self, full_instance=False):
        """
            A slightly modified MaxSAT call. The aim is to reuse previously
            computed cores.
        """

        if full_instance:
            # detecting new unit cores
            self.detect_unit_cores()
        else:
            # using the core-deps oracle to detect and reuse valid cores
            self.reuse_cores()

        # terminate early if we exceed the slack value
        if self.estop and self.cost > self.slack:
            return False

        # main solving loop
        while not self.oracle.solve(assumptions=self.ehard + self.sels + self.sums):
            self.get_core_ext()

            if not self.core:
                # core is empty, i.e. hard part is unsatisfiable
                return False

            # processing the core
            self.process_core_ext()

            if self.verbose > 1:
                print('c cost: {0}; core sz: {1}; soft sz: {2}'.format(self.cost,
                    len(self.core), len(self.sels) + len(self.sums)))

            # terminate early if we exceed the slack value
            if self.estop and self.cost > self.slack:
                return False

        return True

    def reuse_cores(self):
        """
            Detect cores and reuse them using the known core dependencies.
            First, check unit cores and then regular cores.
        """

        # first, unit-size cores
        self.reuse_unit_cores()

        # next, all other cores
        lits = self.cchecker.get_active(self.ehard + self.sels + self.sums)

        # counter of reused cores
        found = 0

        # full assumptions
        assumps = set(self.ehard + self.sels + self.sums)
        for ll in lits:
            core = self.cores[ll]
            for reason in core.reasons:
                # unit cores are already detected;
                # here we expect core.lits to be non-empty
                if core.lits and core.lits | reason <= assumps:
                    # updating the union of all extra hard clauses involved in cores
                    self.reason = self.reason.union(reason)

                    # core weight
                    self.core = list(core.lits)
                    self.minw = min(map(lambda l: self.wght[l], self.core))

                    # dividing the core into two parts
                    iter1, iter2 = itertools.tee(self.core)
                    self.core_sels = list(l for l in iter1 if l in self.sels_set)
                    self.core_sums = list(l for l in iter2 if l not in self.sels_set)

                    # processing the core
                    self.process_core_ext(core_lit=ll)

                    # updating the full list of assumptions
                    assumps = set(self.ehard + self.sels + self.sums)

                    found += 1

                    break

        if self.verbose > 1 and found:
            print('c cores reused:', found)

    def reuse_unit_cores(self):
        """
            Detect and reuse unit-size cores.
        """

        # trying to check active unit cores
        lits = self.cchecker.get_active(self.ehard)

        # garbage literals to collect
        self.garbage = set()

        found = 0

        for l in lits:
            if -l not in self.sels_set:
                continue

            # the literal is active => proceed
            assert self.cores[l].sz == 1, 'Expected unit core, got non-unit!'

            # determining the reason
            reason = set([])
            for r in self.cores[l].reasons:
                if r <= self.ehset:
                    reason = r
                    break

            # updating the global reason
            self.reason = self.reason.union(reason)

            # updating the cost
            self.cost += self.wght[-l]

            # marking as garbage
            self.garbage.add(-l)

            found += 1

        # remove the corresponding assumptions
        self.filter_assumps()

        if self.verbose > 1 and found:
            print('c unit cores reused:', found)

    def detect_unit_cores(self):
        """
            Detect and process unit cores outside of the main loop.
        """

        # assumptions to remove
        self.garbage = set()

        # number of newly detected unit cores
        found = 0

        # checking all available selectors
        for l in self.sels:
            st, props = self.oracle.propagate(assumptions=self.ehard + [l],
                    phase_saving=2)

            if not st:
                # propagating this literal results in a conflict
                # now, we need to attribute responsibility to some
                # of the hard assupmtions
                assert not self.oracle.solve(assumptions=self.ehard + [l])
                reason = []
                for ll in self.oracle.get_core():
                    if ll in self.ehset:
                        reason.append(ll)

                # recording the core for later detection and reuse
                self.record_core(-l, reason=reason)

                # updating the reason
                self.reason = self.reason.union(set(reason))

                # updating the cost
                self.cost += self.wght[l]

                # marking as garbage
                self.garbage.add(l)

                found += 1

        # remove the corresponding assumptions
        self.filter_assumps()

        # updating the set of selectors
        self.sels_set = set(self.sels)

        if self.verbose > 1 and found:
            print('c new unit cores:', found)

    def record_core(self, lsum, tobj=None, tbnd=0, reason=[]):
        """
            Record a new core and its reason.
        """

        # recording it in the core-deps oracle
        if tbnd:
            self.cchecker.record([-ll for ll in reason] + tobj.lits, lsum)
        else:
            self.cchecker.record([-ll for ll in reason], lsum)

        # saving it in the core-info dictionary
        self.cores[lsum].tobj = tobj
        self.cores[lsum].tbnd = tbnd
        self.cores[lsum].lits = frozenset(tobj.lits) if tobj else frozenset()
        self.cores[lsum].sz = 1 if tbnd == 0 else len(tobj.lits)

        # mapping the literals of the core to the totalizer object directly
        self.tots[self.cores[lsum].lits] = tobj

        # adding reason
        reason = set(reason)
        for i, r in enumerate(self.cores[lsum].reasons):
            if r <= reason:
                return
            elif reason < r:
                self.cores[lsum].reasons[i] = reason
                break
        else:
            self.cores[lsum].reasons.append(reason)

        self.cores[lsum].reasons.sort(key=lambda x: len(x))

    def get_core_ext(self):
        """
            Extract unsatisfiable core. The result of the procedure is
            stored in variable ``self.core``. If necessary, core
            trimming and also heuristic core reduction is applied
            depending on the command-line options. A *minimum weight*
            of the core is computed and stored in ``self.minw``.
            Finally, the core is divided into two parts:

            1. clause selectors (``self.core_sels``),
            2. sum assumptions (``self.core_sums``).
        """

        self.core = self.oracle.get_core()

        if self.core:
            # try to reduce the core by trimming
            self.trim_core()

            # filtering out extra hard clauses from the core
            iter1, iter2 = itertools.tee(self.core)
            self.filt = list(l for l in iter1 if l in self.ehset)
            self.core = list(l for l in iter2 if l not in self.ehset)

            # updating the union of all extra hard clauses involved in cores
            self.reason = self.reason.union(set(self.filt))

            # and by heuristic minimization
            self.minimize_core_ext()

            # the core may be empty after core minimization
            if not self.core:
                return

            # core weight
            self.minw = min(map(lambda l: self.wght[l], self.core))

            # dividing the core into two parts
            iter1, iter2 = itertools.tee(self.core)
            self.core_sels = list(l for l in iter1 if l in self.sels_set)
            self.core_sums = list(l for l in iter2 if l not in self.sels_set)

    def minimize_core_ext(self):
        """
            We need to ignore extra hard clauses.
        """

        if self.minz and len(self.core) > 1:
            self.core = sorted(self.core, key=lambda l: self.wght[l])
            self.oracle.conf_budget(1000)

            i = 0
            while i < len(self.core):
                to_test = self.core[:i] + self.core[(i + 1):]

                if not self.oracle.solve_limited(assumptions=self.filt + to_test):
                    self.core = to_test
                else:
                    i += 1

    def process_core_ext(self, core_lit=None):
        """
            The method deals with a core found previously in
            :func:`get_core`. Clause selectors ``self.core_sels`` and
            sum assumptions involved in the core are treated
            separately of each other. This is handled by calling
            methods :func:`process_sels` and :func:`process_sums`,
            respectively. Whenever necessary, both methods relax the
            core literals, which is followed by creating a new
            totalizer object encoding the sum of the new relaxation
            variables. The totalizer object can be "exhausted"
            depending on the option.
        """

        # updating the cost
        self.cost += self.minw

        # assumptions to remove
        self.garbage = set()

        bumped = []

        if len(self.core_sels) != 1 or len(self.core_sums) > 0:
            # process selectors in the core
            self.process_sels()

            # process previously introducded sums in the core
            self.process_sums_ext(bumped, core_lit)

            if len(self.rels) > 1:
                # create a new cardunality constraint
                if not core_lit:
                    rels = frozenset(self.rels)
                    if rels in self.tots:
                        # if literals are known, reusing the old totalizer
                        t = self.tots[rels]
                    else:
                        # otherwise, creating a new one
                        t = self.create_sum()

                    # apply core exhaustion if required
                    b = self.exhaust_core(t) if self.exhaust else 1

                    # recording the new core-dep
                    self.record_core(-t.rhs[b], tobj=t, tbnd=b, reason=self.filt)

                    # recording the bumped sums, i.e lits -> known sum is updated
                    for l in bumped:
                        self.record_core(l, tobj=t, tbnd=b, reason=self.filt)
                else:
                    t = self.cores[core_lit].tobj
                    b = self.cores[core_lit].tbnd

                if b:
                    # save the info about this sum and
                    # add its assumption literal
                    self.set_bound(t, b)
                else:
                    assert 0, 'We expected not to end up here!'
                    # impossible to satisfy any of these clauses
                    # they must become hard
                    for relv in self.rels:
                        self.oracle.add_clause([relv])
        else:
            # unit cores are treated differently
            # (their negation is added to the hard part)
            # self.oracle.add_clause([-self.core_sels[0]])
            self.garbage.add(self.core_sels[0])

        # remove unnecessary assumptions
        self.filter_assumps()

    def process_sums_old(self, bumped, core_lit=None):
        """
            Process cardinality sums participating in a new core.
            Whenever necessary, some of the sum assumptions are
            removed or split (depending on the value of
            ``self.minw``). Deleted sums are marked as garbage and are
            dealt with in :func:`filter_assumps`.

            In some cases, the process involves updating the
            right-hand sides of the existing cardinality sums (see the
            call to :func:`update_sum`). The overall procedure is
            detailed in [1]_.
        """

        # sums that should be deactivated (but not removed completely)
        to_deactivate = set([])

        for l in self.core_sums:
            if self.wght[l] == self.minw:
                # marking variable as being a part of the core
                # so that next time it is not used as an assump
                self.garbage.add(l)
            else:
                # do not remove this variable from assumps
                # since it has a remaining non-zero weight
                self.wght[l] -= self.minw

                # deactivate this assumption and put at a lower level
                # if self.done != -1, i.e. if stratification is disabled
                if self.done != -1 and self.wght[l] < self.blop[self.levl]:
                    self.wstr[self.wght[l]].append(l)
                    to_deactivate.add(l)

            if not core_lit:
                # increase bound for the sum
                t, b = self.update_sum(l)
            else:
                t, b = self.tobj[l], self.bnds[l] + 1

            # updating bounds and weights
            if b < len(t.rhs):
                lnew = -t.rhs[b]
                if lnew in self.garbage:
                    self.garbage.remove(lnew)
                    self.wght[lnew] = 0

                if lnew not in self.wght:
                    self.set_bound(t, b)

                    # if this is not an old (known) core
                    # we need to record the fact of bumping
                    if not core_lit:
                        bumped.append(lnew)
                else:
                    self.wght[lnew] += self.minw

            # put this assumption to relaxation vars
            self.rels.append(-l)

        # deactivating unnecessary sums
        self.sums = list(filter(lambda x: x not in to_deactivate, self.sums))

    def process_sums_new(self, bumped, core_lit=None):
        """
            Process cardinality sums participating in a new core.
            Whenever necessary, some of the sum assumptions are
            removed or split (depending on the value of
            ``self.minw``). Deleted sums are marked as garbage and are
            dealt with in :func:`filter_assumps`.

            In some cases, the process involves updating the
            right-hand sides of the existing cardinality sums (see the
            call to :func:`update_sum`). The overall procedure is
            detailed in [1]_.
        """

        # sums that should be deactivated (but not removed completely)
        to_deactivate = set([])

        for l in self.core_sums:
            if self.wght[l] == self.minw:
                # marking variable as being a part of the core
                # so that next time it is not used as an assump
                self.garbage.add(l)
            else:
                # do not remove this variable from assumps
                # since it has a remaining non-zero weight
                self.wght[l] -= self.minw

                # deactivate this assumption and put at a lower level
                # if self.done != -1, i.e. if stratification is disabled
                if self.done != -1 and self.wght[l] < self.blop[self.levl]:
                    self.wstr[self.wght[l]].append(l)
                    to_deactivate.add(l)

            if not core_lit:
                # increase bound for the sum
                t, b = self.update_sum(l)
            else:
                t, b = self.tobj[l], self.bnds[l] + 1

            # updating bounds and weights
            if b < len(t.rhs):
                lnew = -t.rhs[b]
                if lnew not in self.swgt:
                    self.set_bound(t, b, self.swgt[l])

                    # if this is not an old (known) core
                    # we need to record the fact of bumping
                    if not core_lit:
                        bumped.append(lnew)

            # put this assumption to relaxation vars
            self.rels.append(-l)

        # deactivating unnecessary sums
        self.sums = list(filter(lambda x: x not in to_deactivate, self.sums))

    def get_cost(self, model):
        """
            Given a model, compute its cost.
        """

        cost = 0

        for l, w in self.wght_copy.items():
            if model[abs(l) - 1] == -l:
                cost += w
                if cost > self.slack:
                    break

        return cost

    # def exhaust_core(self, tobj):
    #     """
    #         Core exhaustion augmented with cost check.
    #     """

    #     # the first case is simpler
    #     if self.oracle.solve(assumptions=[-tobj.rhs[1]]):
    #         # if self.estop:
    #         #     self.ubcost = self.get_cost(self.oracle.get_model())
    #         return 1
    #     else:
    #         self.cost += self.minw

    #     for i in range(2, len(self.rels)):
    #         # saving the previous bound
    #         self.tobj[-tobj.rhs[i - 1]] = tobj
    #         self.bnds[-tobj.rhs[i - 1]] = i - 1

    #         # increasing the bound
    #         self.update_sum(-tobj.rhs[i - 1])

    #         if self.oracle.solve(assumptions=[-tobj.rhs[i]]):
    #             # the bound should be equal to i
    #             # if self.estop:
    #             #     self.ubcost = self.get_cost(self.oracle.get_model())
    #             return i

    #         # the cost should increase further
    #         self.cost += self.minw

    #     return None

    # def next_level(self):
    #     """
    #         Here we apply aggressive stratification (in contrast to the
    #         standard stratification).
    #     """

    #     if self.levl >= len(self.blop):
    #         self.levl = None
    #         return

    #     # previously considered weights (if method is called more than once)
    #     wprev = self.blop[self.levl]

    #     while self.levl < len(self.blop) - 1:
    #         # number of selectors with weight less than current weight
    #         numc = sum([len(self.wstr[w]) for w in self.blop[(self.levl + 1):]])

    #         # sum of their weights
    #         sumw = sum([w * len(self.wstr[w]) for w in self.blop[(self.levl + 1):]])

    #         # partial BLO
    #         if self.blop[self.levl] > sumw and sumw != 0:
    #             break

    #         # stratification
    #         if numc / float(len(self.blop) - self.levl - 1) > self.sdiv:
    #             break

    #         # last resort: stratify if the clause weights change a lot
    #         if self.blop[self.levl] / wprev <= 0.9:
    #             break

    #         self.levl += 1
