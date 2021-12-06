#!/usr/bin/env python
##
## experiment.py
##
##  Created on: Aug 27, 2021
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
from options import Options
import os
import random
import shutil
import subprocess
import sys
from xgbooster import XGBooster
import resource


#
#==============================================================================
def parse_options():
    """
        Standard options handling.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:hi:n:r:v',
                ['depth=', 'help', 'inst=',  'num=', 'relax=', 'verbose'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    depth = 5
    inst = 0.3
    num = 50
    relax = 0
    verbose = False

    for opt, arg in opts:
        if opt in ('-d', '--depth'):
            depth = str(arg)
            if depth == 'none':
                depth = -1
            else:
                depth = int(depth)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-i', '--inst'):
            inst = float(arg)
        elif opt in ('-n', '--num'):
            num = int(arg)
        elif opt in ('-r', '--relax'):
            relax = int(arg)
        elif opt in ('-v', '--verbose'):
            verbose = True
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return depth, num, inst, relax, verbose, args


#
#==============================================================================
def usage():
    """
        Prints usage message.
        """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] datasets-list')
    print('Options:')
    print('        -d, --depth=<int>         Tree depth')
    print('                                  Available values: [1 .. INT_MAX], none (default = 5)')
    print('        -h, --help                Show this message')
    print('        -i, --inst=<float,int>    Fraction or number of instances to explain')
    print('                                  Available values: (0 .. 1] or [1 .. INT_MAX] (default = 0.3)')
    print('        -n, --num=<int>           Number of trees per class')
    print('                                  Available values: [1 .. INT_MAX] (default = 50)')
    print('        -r, --relax=<int>         Relax decimal points precision to this number')
    print('                                  Available values: [0 .. INT_MAX] (default = 0)')
    print('        -v, --verbose             Be verbose')


#
#==============================================================================
if __name__ == '__main__':
    depth, num, count, relax, verbose, files = parse_options()

    if files:
        datasets = files[0]
    else:
        datasets = 'datasets.list'

    with open(datasets, 'r') as fp:
        datasets = [line.strip() for line in fp.readlines() if line]

    print(f'training parameters: {num} trees per class, each of depth {"adaptive" if depth == -1 else depth}\n')

    # deleting the previous results
    if os.path.isdir('results'):
        shutil.rmtree('results')
    os.makedirs('results/smt')
    os.makedirs('results/mx')
    # os.makedirs('results/anchor')

    # initializing the seed
    random.seed(1234)

    soptions = Options(f'./xreason.py --relax {relax} -z  -X abd -R lin -u -N 1 -e smt -x \'inst\' somefile'.split())
    moptions = Options(f'./xreason.py --relax {relax} -s g3 -z -X abd -R lin -u -N 1 -e mx -x \'inst\' somefile'.split())

    # training all XGBoost models
    for data in datasets:
        if depth != -1:
            adepth = depth
        else:
            # adaptive length
            data, adepth = data.split()

        print(f'processing {data}...')

        # reading and shuffling the instances
        with open(os.path.join(data), 'r') as fp:
            insts = [line.strip().rsplit(',', 1)[0] for line in fp.readlines()[1:]]
            insts = list(set(insts))
            random.shuffle(insts)

            if count > 1:
                nof_insts = min(int(count), len(insts))
            else:
                nof_insts = min(int(len(insts) * count), len(insts))
            print(f'considering {nof_insts} instances')

        base = os.path.splitext(os.path.basename(data))[0]
        mfile = 'temp/{0}/{0}_nbestim_{1}_maxdepth_{2}_testsplit_0.2.mod.pkl'.format(base, num, adepth)

        slog = open(f'results/smt/{base}.log', 'w')
        mlog = open(f'results/mx/{base}.log', 'w')

        # creating booster objects
        sxgb = XGBooster(soptions, from_model=mfile)
        sxgb.encode(test_on=None)
        mxgb = XGBooster(moptions, from_model=mfile)
        mxgb.encode(test_on=None)

        stimes = []
        mtimes = []
        mcalls = []
        smem = []
        mxmem = []


        #with open("/tmp/texture.samples", 'r') as fp:
        #    insts = [line.strip() for line in fp.readlines()]


        for i, inst in enumerate(insts):
            if i == nof_insts:
                break

            # processing the instance
            soptions.explain = [float(v.strip()) for v in inst.split(',')]
            moptions.explain = [float(v.strip()) for v in inst.split(',')]

            expl1 = sxgb.explain(soptions.explain)

            print(f'i: {inst}', file=slog)
            print(f's: {len(expl1)}', file=slog)
            print(f't: {sxgb.x.time:.3f}', file=slog)
            print('', file=slog)

            smem.append(round(sxgb.x.used_mem / 1024.0, 3))
            stimes.append(sxgb.x.time)

            slog.flush()
            sys.stdout.flush()


            expl2 = mxgb.explain(moptions.explain)
            print(f'i: {inst}', file=mlog)
            print(f's: {len(expl2[0])}', file=mlog)
            print(f't: {mxgb.x.time:.3f}', file=mlog)
            print(f'c: {mxgb.x.calls}', file=mlog)
            print('', file=mlog)
            #
            mxmem.append(round(mxgb.x.used_mem / 1024.0, 3))
            mtimes.append(mxgb.x.time)
            mcalls.append(mxgb.x.calls)
            #
            mxgb.x.calls = 0
            print(f"mem usage: SMT={smem[-1]} MB MaxSAT={mxmem[-1]} MB")

            mlog.flush()
            sys.stdout.flush()

        ##################
        print(f"max time: {max(stimes):.2f}", file=slog)
        print(f"min time: {min(stimes):.2f}", file=slog)
        print(f"avg time: {sum(stimes)/len(stimes):.2f}", file=slog)
        print("", file=slog)

        print(f"max time: {max(mtimes):.2f}", file=mlog)
        print(f"min time: {min(mtimes):.2f}", file=mlog)
        print(f"avg time: {sum(mtimes)/len(mtimes):.2f}", file=mlog)
        print('', file=mlog)
        print(f"avg calls: {sum(mcalls)/len(mcalls):.2f}", file=mlog)
        #################
        mlog.close()
        slog.close()

        print('done')
