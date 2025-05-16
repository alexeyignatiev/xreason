
#
#==============================================================================
from pysat.examples.hitman import *


#
#==============================================================================
class Hitman2(Hitman):

    def add_hard(self, clause, weights=None):
        """
            Add a hard constraint, which can be either a pure clause or an
            AtMostK constraint.

            Note that an optional parameter that can be passed to this method
            is ``weights``, which contains a mapping the objects under
            question into weights. Also note that the weight of an object must
            not change from one call of :meth:`hit` to another.

            :param clause: hard constraint (either a clause or a native AtMostK constraint)
            :param weights: a mapping from objects to weights

            :type clause: iterable(obj)
            :type weights: dict(obj)
        """

        if not len(clause) == 2 or not type(clause[0]) in (list, tuple, set):
            # this is a pure clause
            clause = list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), clause))

            # a soft clause should be added for each new object
            new_obj = filter(lambda vid: abs(vid) not in self.oracle.vmap.e2i, clause)
        else:
            # this is a native AtMostK constraint
            clause = [list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), clause[0])), clause[1]]

            # a soft clause should be added for each new object
            new_obj = filter(lambda vid: abs(vid) not in self.oracle.vmap.e2i, clause[0])

            # there may be duplicate literals if the constraint is weighted
            new_obj = list(set(new_obj))

        # some of the literals may also have the opposite polarity
        new_obj = [l if l in self.idpool.obj2id else -l for l in new_obj]

        # adding the hard clause
        self.oracle.add_clause(clause)

#         if self.htype != 'sat':
#             # new soft clauses
#             for vid in new_obj:
#                 self.oracle.add_clause([-vid], 1 if not weights else weights[self.idpool.obj(vid)])
#         else:
#             # dummy variable id mapping
#             for vid in new_obj:
#                 self.oracle.vmap.e2i[vid] = vid
#                 self.oracle.vmap.i2e[vid] = vid

#             # setting variable polarities
#             self.oracle.set_phases(literals=[self.phase * (-vid) for vid in new_obj])


    def block(self, to_block, weights=None):
        """
            The method serves for imposing a constraint forbidding the hitting
            set solver to compute a given hitting set. Each set to block is
            encoded as a hard clause in the MaxSAT problem formulation, which
            is then added to the underlying oracle.

            Note that an optional parameter that can be passed to this method
            is ``weights``, which contains a mapping the objects under
            question into weights. Also note that the weight of an object must
            not change from one call of :meth:`hit` to another.

            :param to_block: a set to block
            :param weights: a mapping from objects to weights

            :type to_block: iterable(obj)
            :type weights: dict(obj)
        """

        # translating objects to variables
        to_block = list(map(lambda obj: self.idpool.id(obj), to_block))

        # a soft clause should be added for each new object
        new_obj = list(filter(lambda vid: vid not in self.oracle.vmap.e2i, to_block))

        # new hard clause; phase multiplication is needed
        # for making phase switching possible (pure SAT only)
        self.oracle.add_clause([self.phase * (-vid) for vid in to_block])

        # new soft clauses
#         if self.htype != 'sat':
#             for vid in new_obj:
#                 self.oracle.add_clause([-vid], 1 if not weights else weights[self.idpool.obj(vid)])
#         else:
#             # dummy variable id mapping
#             for vid in new_obj:
#                 self.oracle.vmap.e2i[vid] = vid
#                 self.oracle.vmap.i2e[vid] = vid

#             # setting variable polarities
#             self.oracle.set_phases(literals=[self.phase * (-vid) for vid in new_obj])


    def block2(self, to_block, weights=None):
        
        # translating objects to variables
        to_block = list(map(lambda obj: self.idpool.id(obj), to_block))

        # new hard clause; phase multiplication is needed
        # for making phase switching possible (pure SAT only)
        self.oracle.add_clause([self.phase * (vid) for vid in to_block])
        print([self.phase * (vid) for vid in to_block])
        
        
#
#==============================================================================
class Hitman3(Hitman):   
    
    def init(self, bootstrap_with, weights=None, subject_to=[]):

        # formula encoding the sets to hit
        formula = WCNFPlus()

        formula = bootstrap_with.copy()
        for i in range(formula.nv):
            self.idpool.id(i+1)
        
#         # hard clauses
#         for to_hit in bootstrap_with:
#             to_hit = map(lambda obj: self.idpool.id(obj), to_hit)

#             formula.append([self.phase * vid for vid in to_hit])

#         # soft clauses
#         for obj_id in six.iterkeys(self.idpool.id2obj):
#             formula.append([-obj_id],
#                     weight=1 if not weights else weights[self.idpool.obj(obj_id)])
            
#         # additional hard constraints
#         for cl in subject_to:
#             if not len(cl) == 2 or not type(cl[0]) in (list, tuple, set):
#                 # this is a pure clause
#                 formula.append(list(map(lambda obj: self.idpool.id(obj) * (-1 if obj<0 else 1), cl)))
#             else:
#                 # this is a native AtMostK constraint
#                 formula.append([list(map(lambda obj: self.idpool.id(obj) * (-1 if obj<0 else 1), cl[0])), cl[1]], is_atmost=True)


        if self.htype == 'rc2':
            if not weights or min(weights.values()) == max(weights.values()):
                self.oracle = RC2(formula, solver=self.solver, adapt=self.adapt,
                        exhaust=self.exhaust, minz=self.minz, trim=self.trim)
            else:
                self.oracle = RC2Stratified(formula, solver=self.solver,
                        adapt=self.adapt, exhaust=self.exhaust, minz=self.minz,
                        nohard=True, trim=self.trim)
        elif self.htype == 'lbx':
            self.oracle = LBX(formula, solver_name=self.solver,
                    use_cld=self.usecld)
        else:
            raise NotImplementedError("Only rc2 or lbx is supported!")