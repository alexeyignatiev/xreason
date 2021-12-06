import os
import sys

names = [
        'ann-thyroid',
        'appendicitis',
        'biodegradation',
        'divorce',
        'ecoli',
        'glass2',
        'ionosphere',
        'pendigits',
        'promoters',
        'segmentation',
        'shuttle',
        'sonar',
        'spambase',
        'texture',
        'threeOf9',
        'twonorm',
        'vowel',
        'wdbc',
        'wine-recognition',
        'wpbc',
        'zoo'
]


def parse_one(lines):
    stat, inst, time = [], '', None

    for line in lines:
        line = line.strip()

        if line.startswith('i'):
            assert time is None, 'time must be None'
            inst = line[2:]
        elif line.startswith('t'):
            assert inst != '', 'sample is empty'
            time = line[2:]

            stat.append((inst, time))
            inst, time = '', None

    return stat


if __name__ == '__main__':

    print('instance anchor smt mx')

    for name in names:
        with open('anchor/{0}.log'.format(name), 'r') as fp:
            anchor = fp.readlines()
        with open('smt/{0}.log'.format(name), 'r') as fp:
            smt = fp.readlines()
        with open('mx/{0}.log'.format(name), 'r') as fp:
            mx = fp.readlines()

        astat = parse_one(anchor)
        sstat = parse_one(smt)
        mstat = parse_one(mx)

        assert len(astat) == len(sstat) == len(mstat)

        for i, pairs in enumerate(zip(astat, sstat, mstat)):
            assert pairs[0][0] == pairs[1][0] == pairs[2][0], 'samples must be equal'
            print('{0}.{1} {2} {3} {4}'.format(name, i, pairs[0][1], pairs[1][1], pairs[2][1]))
