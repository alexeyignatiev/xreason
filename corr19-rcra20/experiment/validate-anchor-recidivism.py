#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
from anchor_wrap import anchor_call
from data import Data
from options import Options
import os
import resource
import sys
from xgbooster import XGBooster


if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    xgb = XGBooster(options, from_model='../temp/recidivism_data/recidivism_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl')

    # encode it and save the encoding to another file
    xgb.encode()

    with open('../bench/anchor/recidivism/recidivism.samples', 'r') as fp:
        lines = fp.readlines()

    # timers
    atimes = []
    vtimes = []
    ftimes = []
    etimes = []

    tested = set()
    errors = []
    reduced = 0
    for i, s in enumerate(lines):
        options.explain = [float(v.strip()) for v in s.split(',')]

        if tuple(options.explain) in tested:
            continue

        tested.add(tuple(options.explain))
        print('sample {0}: {1}'.format(i, ','.join(s.split(','))))

        # calling anchor
        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        expl = xgb.explain(options.explain, use_anchor=anchor_call)

        print('expl1:', expl)
        print('szex1:', len(expl))

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        atimes.append(timer)

        # validating explanation of anchor
        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        coex = xgb.validate(options.explain, expl)

        timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
        vtimes.append(timer)

        if coex:
            errors.append(1)
            print('incorrect')
            print('   ', coex)

            # fixing explanation of anchor
            timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

            expl2 = xgb.explain(options.explain, expl_ext=expl, prefer_ext=True)

            print('expl2:', expl2)
            print('szex2:', len(expl2))

            timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
            ftimes.append(timer)

            print('fixed: {0} -> {1} ({2} -> {3})'.format(expl, expl2, len(expl), len(expl2)))
        else:
            errors.append(0)
            print('correct')

            # fixing explanation of anchor
            timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

            expl2 = xgb.explain(options.explain, expl_ext=expl)

            print('expl2:', expl2)
            print('szex2:', len(expl2))

            timer = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - timer
            etimes.append(timer)

            if len(expl2) < len(expl):
                reduced += 1
                print('reduced further: {0} -> {1} ({2} -> {3})'.format(expl, expl2, len(expl), len(expl2)))
            else:
                print('failed to reduce')

    print('')
    print('num errors:', sum(errors))
    print('avg errors:', float(sum(errors)) / len(errors))
    print('all samples:', len(lines))
    print('num reduced:', reduced)

    # reporting the time spent
    print('')
    print('tot atime: {0:.2f}'.format(sum(atimes)))
    print('max atime: {0:.2f}'.format(max(atimes)))
    print('min atime: {0:.2f}'.format(min(atimes)))
    print('avg atime: {0:.2f}'.format(sum(atimes) / len(atimes)))
    print('')
    print('tot btime: {0:.2f}'.format(sum(vtimes)))
    print('max btime: {0:.2f}'.format(max(vtimes)))
    print('min btime: {0:.2f}'.format(min(vtimes)))
    print('avg btime: {0:.2f}'.format(sum(vtimes) / len(vtimes)))
    print('')
    print('tot ftime: {0:.2f}'.format(sum(ftimes)))
    print('max ftime: {0:.2f}'.format(max(ftimes)))
    print('min ftime: {0:.2f}'.format(min(ftimes)))
    print('avg ftime: {0:.2f}'.format(sum(ftimes) / len(ftimes)))
    print('')
    print('tot etime: {0:.2f}'.format(sum(etimes)))
    print('max etime: {0:.2f}'.format(max(etimes)))
    print('min etime: {0:.2f}'.format(min(etimes)))
    print('avg etime: {0:.2f}'.format(sum(etimes) / len(etimes)))
