#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## GiXp.py
##
##  Created on: Oct 7, 2024
##      Author: Yacine Izza
##      
##

# ./GiXp.py -v -X abd -M -R lin -e mx -s g3 -x '7.7,3.8,6.7,2.2' ../../models/iris/iris_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl

#
#==============================================================================
from __future__ import print_function

import sys
import numpy as np
from functools import reduce
from six.moves import range
import math

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset
from xgbooster import MXExplainer, MXReasoner

import gurobipy as gp
from gurobipy import GRB
from hitman2 import Hitman3 

#
#==============================================================================

class BoosTree(XGBooster):
    
    def explain(self, sample):
        """
            Compute optimal/max inflated explanation
        """
        if 'x' not in dir(self):
            self.x = IXplainer(self.enc, self.intvs, self.imaps,
                        self.ivars, self.feature_names, self.num_class,
                        self.options, self)
        expl = self.x.explain(np.array(sample), self.options.smallest)

        # returning the explanation
        return expl    
    

class IXplainer(MXExplainer):

    def _explain(self, sample, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, reduce_='none'):
        """
            Compute an explanation.
        """
        
        self.in2Iid = [None]*len(self.sample)
        for h in self.hypos:
            f,v = self.xgb.mxe.vid2fid[h]
            j = self.xgb.mxe.imaps[f][v]
            self.in2Iid[self.v2feat[h]] = j
#         for i,v in enumerate(self.sample): 
#             f = f'{i}'
#             if f in self.xgb.mxe.intvs:
#                 x = next((x for x in self.enc.intvs[f][:-1] if x >= v), self.enc.intvs[f][-1])
#                 j = self.xgb.mxe.imaps[f][x]
#                 self.in2Iid.append(j)                
#             else:
#                 self.in2Iid.append[None]

        if xtype in ('abductive', 'abd'):
            # abductive explanations => MUS computation and enumeration
            if xnum == 1:
                if smallest:
                    iaxp = self._mxmus()
                    axp = list(iaxp.keys())
                else:    
                    axp = self.extract_mus(reduce_=reduce_)
                    iaxp = self.inflateAXp(axp)
                if self.verbose:
                    #print(iaxp)
                    cov = self.coverage(iaxp,log=True)
                    print('  cov:', f"{cov:.2f}")
                    print('  calls:', self.calls)
                self.expls = [axp]
                self.infx = iaxp
            else:
                self.mhs_mus_enumeration(xnum, smallest=smallest)
        else:  # contrastive explanations => MCS enumeration
            if xnum == 1:
                cxp = self.extract_mcs()
                icxp = self.inflateCXp(cxp)
                self.expls = [cxp]
                self.infx = icxp
            else:    
                self.mhs_mcs_enumeration(xnum, smallest, reduce_)
    
    
    def inflateAXp(self, axp):
        """
            Maximal (subset) Inflated AXp
        """
        # self.v2cat: map to local feature of TE
        # self.v2feat: map to original feature or Dataset
        # self.xgb.mxe.vid2fid: map selector to (feature, value) {1: ('f1', 2.75) ...}
        
        hyps = self._cats2hypos(axp)
#         print(hyps)
#         print(self.v2cat, axp, hyps)
#         print()
#         print(self.fcats, self.hypos)
#         print(self.xgb.mxe.vid2fid)
#         print(self.v2feat)
#         print()
#         print( self._cats2hypos([self.v2cat[s] for s in self.v2cat]) )        
        
        #iaxp =  {i:[h] for i,h in zip(axp, hyps)}
        iAXp =  {self.v2feat[h]:[h] for i,h in zip(axp, hyps)}
        
        for i,h in enumerate(hyps): 
            to_test = hyps[i+1:] # + hyps[:i]
            f = self.xgb.mxe.vid2fid[h][0]
            fid = self.v2feat[h]
            domain = self.ivars[f]
            l,u = domain.index(h)-1, domain.index(h)+1
            # inflate lower bound   
            while (l >= 0) and not self.oracle.get_coex(to_test+[domain[l]], early_stop=True):
                # iAXp[fid].append(domain[l])
                iAXp[fid] = [domain[l]] + iAXp[fid]
                l -= 1  
                self.calls += 1            
            # inflate upper bound
            while (u < len(domain)) and not self.oracle.get_coex(to_test+[domain[u]], early_stop=True):
                iAXp[fid].append(domain[u])
                u += 1
                self.calls += 1
                
            # add clause
            if len(iAXp[fid]) > 1: # feature has been inflated
                # (x_i = v_i) or (x_i=v_j) or....(x_i=v_n)
                cl = iAXp[fid]
                assert (self.oracle.ortype == 'int') # incremental MaxSAT mode
                for clid in self.oracle.oracles: 
                    self.oracle.oracles[clid].add_clause(cl) 
                    # self.oracle.formulas[clid].hard    
        
        # map lits to interval indices        
        iAXp = {i:sorted([self.xgb.mxe.imaps[self.xgb.mxe.vid2fid[h][0]][self.xgb.mxe.vid2fid[h][1]] \
                                                               for h in iAXp[i]]) for i in iAXp}
        return iAXp
    

    def inflateCXp(self, cxp):
        """
            Maximal (subset) Inflated CXp
        """
        to_test = self._cats2hypos(cxp)  
        hyps = [h for h in self.hypos if (h not in to_test)]
        hyps += [-h for h in to_test]
        iCXp = {}
        #to_test = [i for i in self.v2feat if (i not in cxp)]
        
        for h in to_test: 
            fid = self.v2feat[h]
            iCXp[fid] = [] 
            f = self.xgb.mxe.vid2fid[h][0]
            domain = self.ivars[f]
            
            if len(domain) == 2: # binary
                iCXp[fid] = [domain[0] if (domain[0]!=h) else domain[1]]
                
            for t in domain:
                if t == h:
#                     if len(iCXp[fid]):
#                         break
#                     else:    
#                         continue
                    continue
                if self.oracle.get_coex(hyps+[t], early_stop=True):
                    iCXp[fid].append(t)
#                 elif len(iCXp[fid]):
#                     # cannot enlarge interval anymore, stop
#                     break 
        #print(iCXp)
        return iCXp
    
    
    
    def _mxmus(self):
        """
            compute a cardinality-maximal Inflated AXp
        """
        save_file_name = './temp/grb.lp'
        grb =  gp.Model(os.path.basename(save_file_name))
        grb.setParam(GRB.Param.OutputFlag, 0)
        grb.setParam(GRB.Param.LogToConsole, 0)
        
        enc = self.xgb.mxe
        formula = self.xgb.enc
        x_min, x_max = np.min(self.xgb.X, axis=0), np.max(self.xgb.X, axis=0) 
        
        if self.optns.encode == 'mxa':
            ortype = 'alien'
        elif self.optns.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'   
        
        #=======================================#
        def newVar(vname: str):
            """
                If a variable named 'name' already exists then
                return its id; otherwise create a new var
            """
            if vname in enc.idmgr.obj2id: #var has been already created 
                l = enc.idmgr.obj2id[vname]
            l = enc.idmgr.id(vname)
            return l
        #=======================================#
        
        
        to_hit, subject_to = [], []
        weights = []
        for h in self.hypos: # h,i in self.v2cat.items()
            f, i = enc.vid2fid[h][0], self.v2feat[h]
            #assert (int(f[1:]) == i)
            splits = [x_min[i]] + enc.intvs[f][:-1] + [x_max[i]]
            vars, wght = [], []
            for j in range(self.in2Iid[i]+1):
                for k in range(self.in2Iid[i],len(enc.ivars[f])):
                    y = grb.addVar(vtype=GRB.BINARY, name=f'f{i}_l{j}_u{k}')
                    w = math.log(splits[k+1] - splits[j])
                    vars.append(y)
                    wght.append(w) # subject_to maximze
            to_hit.append(vars)
            grb.addLConstr(gp.LinExpr([1]*len(vars), vars), GRB.EQUAL, 1)
            subject_to.extend(vars)
            weights.extend(wght)
            
        # Set objective
        grb.setObjective(gp.LinExpr(weights, subject_to), GRB.MAXIMIZE)

        fid2Sid = {self.v2feat[h]:i for i,h in enumerate(self.hypos)}
        self.calls = 0
        
        # encode lower and upper bounds
        LBs, UBs = {}, {}
        for _,fid in self.v2feat.items():
            f = "f{0}".format(fid)
            lb = [grb.addVar(vtype=GRB.BINARY, name=f'l{fid}>T{j}') for j in range(self.in2Iid[fid])]
            ub = [grb.addVar(vtype=GRB.BINARY, name=f'u{fid}<T{j}') for j in range(self.in2Iid[fid],len(enc.intvs[f])-1)]   
            LBs[fid], UBs[fid] = lb, ub
                
            for j in range(len(lb)-1):
                grb.addLConstr(lb[j+1] - lb[j], GRB.LESS_EQUAL, 0) # e.g. [li ≥ 0.2] → [li ≥ −0.4]
            for j in range(len(ub)-1):
                grb.addLConstr(ub[j] - ub[j+1], GRB.LESS_EQUAL, 0) # e.g. [ui < −0.4] → [ui < 0.2]
            
                
            for j in range(self.in2Iid[fid]+1):
                r = len(enc.ivars[f]) - self.in2Iid[fid]
                for k in range(r):
                    y = to_hit[fid2Sid[fid]][j*r+k]    
                    coefs, vars = [], []
                    #lits = []
                    if j == 0:
                        if j<self.in2Iid[fid]:
                            #lits.append(-lb[j])
                            vars.append(lb[j])
                            coefs.append(-1)
                    elif j < self.in2Iid[fid]:
                        #lits.append(lb[j-1])
                        #lits.append(-lb[j])
                        vars.append(lb[j-1])
                        vars.append(lb[j])
                        coefs.append(1)
                        coefs.append(-1)
                    else: # j == self.in2Iid[fid]
                        #lits.append(lb[j-1])
                        vars.append(lb[j-1])
                        coefs.append(1)
                    if k == 0:
                        if k < r-1:
                            #lits.append(ub[k])
                            vars.append(ub[k])
                            coefs.append(1)
                    elif k < r-1:
                        #lits.append(-ub[k-1])
                        #lits.append(ub[k])
                        vars.append(ub[k-1])
                        vars.append(ub[k])
                        coefs.append(-1)
                        coefs.append(1)
                    else:
                        #lits.append(-ub[k-1])
                        vars.append(ub[k-1])
                        coefs.append(-1)
                        
                    #assert len(lits)
                    #formula.append([y]+[-l for l in lits])
                    #formula.extend([[-y, l] for l in lits])
                    
                    rhs = len(coefs) - 1 - len([c for c in coefs if c<0])
                    grb.addLConstr(gp.LinExpr(coefs+[-1], vars+[y]), GRB.LESS_EQUAL, rhs)
                    for c,x in zip(coefs, vars):
                        rhs = 1 if c<0 else 0
                        grb.addLConstr(gp.LinExpr([1, -c], [y, x]), GRB.LESS_EQUAL, rhs)
        
        #grb.write('temp/grb.lp') 
        grb.update()
        """
        grb.optimize()
        print('runtime:', grb.Runtime)
        if grb.Status == GRB.OPTIMAL: # get predicted class
            print([(v.VarName, v.X) for hs in to_hit for v in hs if v.X>0]) 
            print('Obj: %g' % grb.ObjVal)    
        else:
            assert (grb.Status == GRB.INFEASIBLE)
            print(grb.Status)
        """
        #==================================================================================#    
        # compute CXp's of size 1 if any exists
        for i,h in enumerate(self.hypos):
            f, fid = enc.vid2fid[h][0], self.v2feat[h]
            to_test = self.hypos[:i]+self.hypos[i+1:]
            aex = self.oracle.get_coex(to_test, early_stop=True)
            if aex:
                intvs = enc.ivars[f]
                icxp = [j for j,p in enumerate(intvs) if (aex[abs(p)-1] == p)] # same sign literal p and model 
                assert len(icxp) == 1
                assert (icxp[0] != self.in2Iid[fid])
                l, u = icxp[0]-1, icxp[0]+1
                # inflate upper bound
                while (u < self.in2Iid[fid]) and self.oracle.get_coex(to_test+[intvs[u]], early_stop=True):
                    icxp.append(u)
                    u += 1
                # inflate lower bound    
                while (l > self.in2Iid[fid]) and self.oracle.get_coex(to_test+[intvs[l]], early_stop=True):
                    icxp.append(l)
                    l -= 1
                icxp.sort()

                k = len(intvs)-self.in2Iid[fid]        
                if icxp[0] > self.in2Iid[fid]:
                    to_block = [to_hit[i][j*k+(icxp[0]-self.in2Iid[fid])+u] for j in range(self.in2Iid[fid]+1) 
                                    for u in range(len(intvs) - icxp[0])]
                    b = UBs[fid][icxp[0]-self.in2Iid[fid]-1]
                else:
                    assert icxp[-1] < self.in2Iid[fid] 
                    to_block = to_hit[i][:(icxp[-1]+1)*k]
                    ##
                    b= LBs[fid][icxp[-1]]

                grb.addLConstr(b, GRB.GREATER_EQUAL, 1)
                    
                if self.verbose > 1:    
                    print('to_block:', b.VarName) 
        #==================================================================================# 
        
        # Counterexample-guided abstraction refinement (CEGAR) loop
        self.calls, otime = 0, 0.
        while True:
            grb.optimize()
            assert (grb.Status == GRB.OPTIMAL)
            hset = [v for hs in to_hit for v in hs if v.X>0]  
            
            #assert len(hset) == len(self.hypos)
            if self.verbose > 1:
                print('\nhset:', [p.VarName for p in hset])
            self.calls += 1
            #print(grb.Runtime, grb.ObjVal)
            otime += grb.Runtime 
                
            hyps, iAXp = [], {}
            for i,p in enumerate(hset): # same order as hypos
                f,j,k = p.VarName.split('_')
                fid, j, k = int(f[1:]), int(j[1:]), int(k[1:])

                if k-j+1 < len(enc.ivars[f]):
                    cl = enc.ivars[f][j:k+1]
                    for clid in self.oracle.oracles: 
                        self.oracle.oracles[clid].add_clause(cl)
                    iAXp[fid] = list(range(j,k+1))
                #else: # free f_i
                #    hyps.append(-self.hypos[i])
            
            aex = self.oracle.get_coex(hyps, early_stop=True)
            if aex is not None:
                model = aex
                to_test = [model[abs(h)-1] for h in self.hypos]
                # reduce WCXp to CXp
                for i,h in enumerate(to_test):
                    if h != self.hypos[i]: # different sign
                        to_test[i] = -h
                        aex = self.oracle.get_coex(to_test+hyps, early_stop=True)
                        if aex:
                            model = aex # save last sat assignment          
                        else:
                            to_test[i] = h
                # get cxp
                icxp, hyps = {}, []
                for i,h in enumerate(to_test): 
                    if h != self.hypos[i]: # free features
                        #f = "f{0}".format(self.v2feat[h])
                        f, fid = enc.vid2fid[-h][0], self.v2feat[-h] # flip sign of h
                        for j,I in enumerate(enc.ivars[f]):
                            if model[abs(I)-1] == I: # same sign
                                icxp[fid] = [j]
                                break
                    else:
                        # fixed features
                        hyps.append(self.hypos[i])

                #print(icxp)
                to_test = []
                # inflate CXp 
                for fid in icxp:
                    intvs = enc.ivars[f'f{fid}']
                    l, u = icxp[fid][0]-1, icxp[fid][0]+1
                    # inflate upper bound
                    while (u < self.in2Iid[fid]) and self.oracle.get_coex(hyps+to_test+[intvs[u]], early_stop=True):
                        icxp[fid].append(u)
                        u += 1
                    # inflate lower bound    
                    while (l > self.in2Iid[fid]) and self.oracle.get_coex(hyps+to_test+[intvs[l]], early_stop=True):
                        icxp[fid].append(l)
                        l -= 1
                    if (icxp[fid][0] < self.in2Iid[fid] and  len(icxp[fid]) < len(intvs[:self.in2Iid[fid]])) or \
                        (icxp[fid][0] > self.in2Iid[fid] and len(icxp[fid]) < len(intvs[self.in2Iid[fid]+1:])):
                        cl =  [intvs[j] for j in icxp[fid]]
                        for clid in self.oracle.oracles: 
                            self.oracle.oracles[clid].add_clause(cl)    
                    else:
                        to_test.append(intvs[icxp[fid][0]]) # single interval, no expansion
                    icxp[fid].sort() 
                    # icxp[fid] = [icxp[fid][0]] + ([icxp[fid][-1]] if (icxp[fid][-1] > icxp[fid][0]) else []) 
                    icxp[fid] = (icxp[fid][0], icxp[fid][-1]) # (lb, ub)

                #print(icxp)
                to_block = []
                # block bounds                    
                for fid in icxp:
                    if (icxp[fid][-1] < self.in2Iid[fid]):
                        b = LBs[fid][icxp[fid][-1]]  
                    else: 
                        assert (icxp[fid][0] > self.in2Iid[fid])
                        b = UBs[fid][icxp[fid][0]-self.in2Iid[fid]-1]

                    to_block.append(b)

                if self.verbose > 1:    
                    print('to_block:', [p.VarName for p in to_block])
                
                #hitman.oracle.add_clause(to_block)
                grb.addLConstr(gp.LinExpr([1]*len(to_block), to_block), GRB.GREATER_EQUAL, 1)
                
            else:
                # optimal solution (MxIAXp) found
                # end of cegar loop
                break
            
            self.oracle.delete()
            # no incremental mode in IHS loop, core cache crashes
            self.oracle = MXReasoner(formula, self.out_id,
                    solver=self.optns.solver,
                    oracle=ortype,
                    am1=self.optns.am1, exhaust=self.optns.exhaust,
                    minz=self.optns.minz, trim=self.optns.trim)             
            
        if self.verbose:
            print('oracle time:', f'{otime:.3f}')
            
        return iAXp
                     

    def coverage(self, infx, log=False):
        x_min, x_max = np.min(self.xgb.X, axis=0), np.max(self.xgb.X, axis=0)
        intvs = self.xgb.mxe.intvs
        feats = [self.v2feat[h] for h in self.hypos]
        if log:
            cov = 0.
            x = x_max - x_min
            for i in range(len(x)):
                if not x[i]:
                    x[i] = 1
            topw = np.log(x)
            #f, fid = enc.vid2fid[h][0], self.v2feat[h]
            for i in infx:
                f = "f{0}".format(i)
                splits = [x_min[i]] + intvs[f][:-1] + [x_max[i]]
                cov += math.log(splits[infx[i][-1]+1] - splits[infx[i][0]])   
            lowc = 0.
            for i in feats:
                f = "f{0}".format(i)
                splits = [x_min[i]] + intvs[f][:-1] + [x_max[i]]
                lowc += math.log(splits[self.in2Iid[i]+1] - splits[self.in2Iid[i]]) 
            cov -= lowc            
            cov += sum([topw[i] for i in feats if (i not in infx)])
            cov /= sum([topw[i] for i in feats]) - lowc
        else:
            cov = 1.
            for i in infx:
                f = "f{0}".format(i)
                splits = [x_min[i]] + intvs[f][:-1] + [x_max[i]]
                cov *= (splits[infx[i][-1]+1] - splits[infx[i][0]]) / (x_max[i] - x_min[i])
        
        return cov*100.0

#
#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    

    if (options.preprocess_categorical):
        preprocess_dataset(options.files[0], options.preprocess_categorical_files)
        exit()

    assert len(options.files)
    xgb = None

    if options.use_categorical:
        data = Data(filename=options.files[0], mapfile=options.mapfile,
                separator=options.separator,
                use_categorical = options.use_categorical)
        
        xgb = XGBooster(options, from_data=data)
        train_accuracy, test_accuracy, model = xgb.train()

    # read a sample from options.explain
    if options.explain:
        options.explain = [float(v.strip()) for v in options.explain.split(',')]
        

    if options.explain:
        if not xgb:
            # abduction-based approach requires an encoding
            xgb = BoosTree(options, from_model=options.files[0])
        
        # encode 
        xgb.encode()
        
        # explain using abduction-based approach
        expl = xgb.explain(options.explain)
            
