#!/bin/sh

# validating anchor
./validate-anchor-adult.py -e smt -s z3 -c -v temp/adult_data/adult_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > adult-validated.anchor

./validate-anchor-lending.py -e smt -s z3 -c -v temp/lending_data/lending_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > lending-validated.anchor

./validate-anchor-recidivism.py -e smt -s z3 -c -v temp/recidivism_data/recidivism_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > recidivism-validated.anchor

./validate-anchor-compas.py -e smt -s z3 -c -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > compas-validated.anchor

./validate-anchor-german.py -e smt -s z3 -c -v temp/german_data/german_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > german-validated.anchor

# validating lime
./validate-lime-adult.py -e smt -s z3 -c -v temp/adult_data/adult_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > adult-validated.lime

./validate-lime-lending.py -e smt -s z3 -c -v temp/lending_data/lending_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > lending-validated.lime

./validate-lime-recidivism.py -e smt -s z3 -c -v temp/recidivism_data/recidivism_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > recidivism-validated.lime

./validate-lime-compas.py -e smt -s z3 -c -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > compas-validated.lime

./validate-lime-german.py -e smt -s z3 -c -v temp/german_data/german_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > german-validated.lime

# validating shap
./validate-shap-adult.py -e smt -s z3 -c -v temp/adult_data/adult_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > adult-validated.shap

./validate-shap-lending.py -e smt -s z3 -c -v temp/lending_data/lending_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > lending-validated.shap

./validate-shap-recidivism.py -e smt -s z3 -c -v temp/recidivism_data/recidivism_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > recidivism-validated.shap

./validate-shap-compas.py -e smt -s z3 -c -v temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > compas-validated.shap

./validate-shap-german.py -e smt -s z3 -c -v temp/german_data/german_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl > german-validated.shap
