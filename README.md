# XReason (former XPlainer)

## (extended to support the MaxSAT-based approach)

This repository contains the tool accompanying a recent AAAI'22 paper on *"Using MaxSAT for Efficient Explanations of Tree Ensembles"*. This version of the `XReason` tool builds on the [earlier original](https://github.com/alexeyignatiev/xreason/corr19-rcra20/experiment/), which aims at reasoning about explanations for machine learning models. The original implementation makes use of the SMT encoding and so an SMT-based explainer. This work extends `XReason` to support the MaxSAT-based explanation extraction approach proposed in the AAAI'22 paper.

Concretely, the modification includes several additional files as well as a few modified original modules, aiming at adding support for a novel MaxSAT-based explainer of boosted tree models. For instance, the encoder now has a class producing a MaxSAT encoding proposed in the paper while the explainer contains the necessary procedures for invoking an incremental MaxSAT-based entailment oracle. The newly added files include `erc2.py`, which extends the RC2 MaxSAT solver with the necessary incrementality capabilities, and `mxreason.py`, which instruments the calls to ERC2.

Note that the original capabilities of `XReason` to reason about heuristic explanations should still be functional.

## Getting Started

The following packages are necessary to run XReason:

* [Anchor](https://github.com/marcotcr/anchor) (version [0.0.2.0](https://pypi.org/project/anchor-exp/0.0.2.0/))
* [anytree](https://anytree.readthedocs.io/)
* [LIME](https://github.com/marcotcr/lime)
* [namedlist](https://gitlab.com/ericvsmith/namedlist)
* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pySMT](https://github.com/pysmt/pysmt) (with Z3 installed)
* [PySAT](https://github.com/pysathq/pysat)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SHAP](https://github.com/slundberg/shap)
* [XGBoost](https://github.com/dmlc/xgboost/) (version [1.7.5](https://pypi.org/project/xgboost/1.7.5/))

**Important:** If you are using a MacOS system, please make sure you use `libomp` (OpenMP) version 11. Later versions are affected by [this bug](https://github.com/dmlc/xgboost/issues/7039).

## Usage

XReason has a number of parameters, which can be set from the command line. To see the list of options, run (the executable script is located in [src](./src)):

```
$ xreason.py -h
```

### Preparing a dataset

**Important:** if a dataset contains continuous data, you can omit this stage and use XReason straight away (with no option ```-c``` specified). Otherwise, read this part.

XReason can be used with datasets in the CSV format. Otherwise, you need to process the categorical features of the dataset. For this, you need to do a few steps:

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
$ xreason.py -p --pfiles compas.csv,compas ../corr19-rcra20/bench/fairml/compas/
```

### Training a tree ensemble

Before extracting explanations, an XGBoost model must be trained:

```
$ xreason.py -c -t -n 50 bench/fairml/compas/compas_data.csv
```

Here, 50 trees per class are trained. Also, parameter ```-c``` is used because the data is categorical. We **emphasize** that parameter ```-c``` should not be used for continuous data, which is the case in the experimental results we obtained in the paper. By default, the trained model is saved in the file ```temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl```.

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


### Computing a formal SMT-based explanation

A rigorous logic-based explanation for the same data instance can be computed by running the following command:

```
$ xreason.py -c -e smt -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Here, parameter ```-e``` specifies the model encoding (SMT) while parameter ```-s``` identifies an SMT solver to use (various SMT solvers can be installed in [pySMT](https://github.com/pysmt/pysmt) - here we use [Z3](https://github.com/Z3Prover/z3)). This command computes a *subset-minimal* explanation, i.e. it is guaranteed that *no proper subset* of the reported explanation can serve as an explanation for the given prediction. Once again, parameter ```-c``` **should be ommited** in the case of continuous data.

Alternatively, a *cardinality-minimal* (i.e. smallest size) explanation can be computed by specifying the ```-M``` option additionally:

```
$ xreason.py -c -e smt -M -s z3 -x '5,0,0,0,0,0,0,0,0,0,0' temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

### Computing a formal MaxSAT-based explanation

Abductive explanations with the use of MaxSAT can be computed in the following way:

```
$ xreason.py -c -X abd -R lin -e mx -s g3 -x '5,0,0,0,0,0,0,0,0,0,0' -vv temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
```

Here, parameter ```-e``` serves to specify the MaxSAT encoding while parameter ```-s``` identifies an underlying SAT solver to use ([PySAT](https://pysathq.github.io/) supports various SAT solvers - here we use Glucose 3). Just like with the case of SMT, this command computes a *subset-minimal* abductive explanation. Parameter `-R` specifies how explanations should be reduced. Iin this case, linear search is used. For QuickXPlain, use `-R qxp`. Once again, parameter ```-c``` **should be ommited** in the case of continuous data.

Alternatively, a *cardinality-minimal* (i.e. smallest size) abductive explanation can be computed by specifying the ```-M``` option additionally:

```
$ xreason.py -c -X abd -M -e mx -s g3 -x '5,0,0,0,0,0,0,0,0,0,0' -vv temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl
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

Also, see [a few example scripts](corr19-rcra20/experiment) for details on how to validate heuristic explanations for every unique sample of the benchmark datasets (note that each of the datasets must be properly processed and the corresponding models must be trained in advance).

## Reproducing experimental results of the AAAI'22 paper

We have carefully prepared the experimental setup so that anyone interested could reproduce our results reported in the paper. To be able to do so, the following steps should be done:

1. Go to the `src` directory:

```
$ cd src/
```

2. Train all the 21 models at once:

```
$ ./aaai22-train-all.py -d none
```

3. Given the trained models, run the experimentation script:

```
$ ./aaai22-experiment.py -i 200 -d none
```

As indicated above, the final command will run the experiment the way it is set up for the paper. (**Note** that this will take a while.) The script invokes `XReason ` with all the necessary parameters set. It traverses the 21 trained models and then randomly picks at most 200 data instances (the random seed is fixed to ensure reproducibility) and invokes (1) the SMT-based explainer followed by (2) the MaxSAT-based explainer. While doing so, it collects the necessary data including explanation size, running time, memory used, et cetera. All the results will be saved in the `results` directory.

### On additional experiments with Anchor for AAAI'22

As one of the reviewers requested *additional* experimental results on comparing our approach against a model-agnostic explainer, we performed an additional experimental setup for doing so. We provide a Notebook script "aaai22_exp_anchor.ipynb" used to run these additional experiments for Anchor.

Note that, for our experiments we used  the most recent version of Anchor (version [0.0.2.0](https://pypi.org/project/anchor-exp/0.0.2.0/)).


### Models used in the AAAI'22 paper

Note that all datasets and the corresponding models are additionally provided in [bench](aaai22/bench) and [models](aaai22/models), respectively. Use may want to opt to use our original models instead of training your own from scratch (see step 2 above). In order to do so, replace step 2 above with the following:

```
$ rm -r src/temp
$ cp -r aaai22/models src/temp
```

### Logs of the AAAI'22 experiments and results

All the logs we obtained in our experiments can be found in the [logs](aaai22/logs) directory. The full table of results can be found in [tables](aaai22/tables).

## Reproducing experimental results of CoRR'19-RCRA'20

Although it seems unlikely that the experimental results reported in the RCRA'20 paper can be reproduced (due to *randomization* used in the training phase), similar results can be obtained if the following commands are executed:

```
$ cd corr19-rcra20/experiment/
$ ./train-all.sh && ./extract-samples.sh
$ ./validate-all.sh
```

The final command should run the experiment the way it was set up for the paper. (**Note** that this will take a while.) The result files will contain the necessary statistics.

## Citations

If `XReason` has been significant to a project that leads to an academic publication, please, acknowledge that fact by citing it:

```
@article{inms-corr19,
  author    = {Alexey Ignatiev and
               Nina Narodytska and
               Joao Marques{-}Silva},
  title     = {On Validating, Repairing and Refining Heuristic {ML} Explanations},
  journal   = {CoRR},
  volume    = {abs/1907.02509},
  year      = {2019}
}

@inproceedings{inms-rcra20,
  author    = {Alexey Ignatiev and
               Nina Narodytska and
               Joao Marques{-}Silva},
  title     = {On Formal Reasoning about Explanations},
  booktitle = {RCRA},
  year      = {2020},
}

@inproceedings{iisms-aaai22a,
  author    = {Alexey Ignatiev and
               Yacine Izza and
               Peter J. Stuckey and
               Joao Marques-Silva},
  title     = {Using MaxSAT for Efficient Explanations of Tree Ensembles},
  booktitle = {AAAI},
  year      = {2022},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
