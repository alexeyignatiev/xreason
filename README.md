# XReason (former XPlainer)

A set of Python scripts for reasoning about explanations of machine learning models. Concretely, the implementation targets tree ensembles trained with [XGBoost](https://xgboost.ai/) and supports computing subset- and cardinality-minimal *rigorous* explanations using on the [abduction-based approach](https://arxiv.org/abs/1811.10656) proposed recently.

## Getting Started

Before using XReason, make sure you have the following Python packages installed:

* [Anchor](https://github.com/marcotcr/anchor)
* [anytree](https://anytree.readthedocs.io/)
* [LIME](https://github.com/marcotcr/lime)
* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pySMT](https://github.com/pysmt/pysmt)
* [PySAT](https://github.com/pysathq/pysat)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SHAP](https://github.com/slundberg/shap)
* [XGBoost](https://github.com/dmlc/xgboost/)

Please, follow the installation instructions on these projects' websites to install them properly. (If you spot any other package dependency not listed here, please, let us know.)

## Usage

XReason has a number of parameters, which can be set from the command line. To see the list of options, run:

```
$ xreason.py -h
```

### Preparing a dataset

XReason can be used with datasets in the CSV format. If a dataset contains continuous data, you can use XReason straight away (with no option ```-c``` specified). Otherwise, you need to process the categorical features of the dataset. For this, you need to do a few steps:

1. Assume your dataset is stored in file ```somepath/dataset.csv```.
2. Create another file named ```somepath/dataset.csv.catcol``` that contains the indices of the categorical columns of ```somepath/dataset.csv```. For instance, if columns ```0```, ```1```, and ```5``` contain categorical data, the file should contain the lines

	```
	0
	1
	5
	```

3. Now, the following command:

```
$ xreason.py -p --pfiles dataset.csv,somename somepath/
```

creates a new file ```somepath/somename_data.csv``` with the categorical features properly handled. As an example, you may want to check the command on the [benchmark datasets](bench), e.g.

```
$ xreason.py -p --pfiles compas.csv,compas bench/fairml/compas/
```

### Training a tree ensemble

Before extracting explanations, an XGBoost model must be trained:

```
$ xreason.py -c -t -n 50 bench/fairml/compas/compas_data.csv
```

Here, 50 trees per class are trained. Also, parameter ```-c``` is used because the data is categorical. By default, the trained model is saved in the file ```temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl```.

### Computing a heuristic explanation

Heuristic explanations can be computed using either parameter ```-q``` ```-l``` for a data instance specified with option ```-x 'comma,separated,values'```. For example, given an instance ```5,0,0,0,0,0,0,0,0,0,0``` and the trained model, the prediction for this instance can be explained by Anchor like this:

```
$ xreason.py -c -q -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

If you want to compute an explanation using LIME, execute the following command. Note that LIME computes an explanation of a size specified as input, using option ```-L```. In this example we instruct LIME to use 5 feature values in the explanation:

```
$ xreason.py -c -l -L 5 -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

The prediction for this instance can be explained by SHAP like this:

```
$  xreason.py -c -w -L 5 -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Note that XReason can also restrict explanations of SHAP to be of a given size (see option ```-L```), which is *not done* by default.

### Computing a rigorous explanation

A rigorous logic-based explanation for the same data instance can be computed by running the following command:

```
$ xreason.py -c -e smt -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Here, parameter ```-e``` specifies the model encoding (SMT) while parameter ```-s``` identifies an SMT solver to use (various SMT solvers can be installed in [pySMT](https://github.com/pysmt/pysmt) - here we use [Z3](https://github.com/Z3Prover/z3)). This command computes a *subset-minimal* explanation, i.e. it is guaranteed that *no proper subset* of the reported explanation can serve as an explanation for the given prediction.

Alternatively, a *cardinality-minimal* (i.e. smallest size) explanation can be computed by specifying the ```-M``` option additionally:

```
$ xreason.py -c -e smt -M -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

### Validating heuristic explanations

XReason can be used to *validate* heuristic explanations computed by LIME, Anchor, or SHAP. Moreover, they can be *repaired* (*refined* further, resp.) if they are invalid (valid, resp.). Although there is no command line setup for doing this, a simple Python script can be devised for doing so:

```python
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

    # load a trained model
    xgb = XGBooster(options, from_model='temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl')

    # encode it and save the encoding to another file
    xgb.encode()

    with open('bench/anchor/compas/compas_data.csv', 'r') as fp:
        lines = fp.readlines()
        del(lines[0])

    tested = set()
    for i, s in enumerate(lines):
        options.explain = [float(v.strip()) for v in s.split(',')[:-1]]

        if tuple(options.explain) in tested:
            continue

        tested.add(tuple(options.explain))
        print('sample {0}: {1}'.format(i, ','.join(s.split(','))))

        # calling anchor
        texpl = xgb.explain(options.explain, use_anchor=anchor_call)
        print('target expl:', texpl)

        # validating the explanation
        coex = xgb.validate(options.explain, texpl)

        if coex:
            print('incorrect (a counterexample exits)\n   ', coex)

            # repairing the target explanation
            rexpl = xgb.explain(options.explain, expl_ext=expl, prefer_ext=True)

            print('rigorous expl (repaired):', rexpl)
        else:
            print('correct')

            # an attempt to refine the target explanation further
            rexpl = xgb.explain(options.explain, expl_ext=expl)

            print('rigorous expl (refined):', rexpl)
```

Also, see [a few example scripts](experiment) for details on how to validate heuristic explanations for every unique sample of the benchmark datasets (note that each of the datasets must be properly processed and the corresponding models must be trained in advance).

## Reproducing experimental results

Although it seems unlikely that the experimental results reported in the paper can be reproduced (due to *randomization* used in the training phase), similar results can be obtained if the following commands are executed:

```
$ cd experiment/
$ ./train-all.sh && ./extract-samples.sh
$ ./validate-all.sh
```

The final command should run the experiment the way it was set up for the paper. (**Note** that this will take a while.) The result files will contain the necessary statistics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
