#!/bin/sh

cd ../

# adult
./xplainer.py -p --pfiles adult.csv,adult bench/anchor/adult/
./xplainer.py -c -t -n 50 bench/anchor/adult/adult_data.csv

# lending
./xplainer.py -p --pfiles lending.csv,lending bench/anchor/lending/
./xplainer.py -c -t -n 50 bench/anchor/lending/lending_data.csv

# recidivism
./xplainer.py -p --pfiles recidivism.csv,recidivism bench/anchor/recidivism/
./xplainer.py -c -t -n 50 bench/anchor/recidivism/recidivism_data.csv

# compas
./xplainer.py -p --pfiles compas.csv,compas bench/fairml/compas/
./xplainer.py -c -t -n 50 bench/fairml/compas/compas_data.csv

# german
./xplainer.py -p --pfiles german.csv,german bench/fairml/german/
./xplainer.py -c -t -n 50 bench/fairml/german/german_data.csv
