#!/bin/sh

# adult
./xreason.py -p --pfiles adult.csv,adult ../bench/anchor/adult/
./xreason.py -c -t -n 50 ../bench/anchor/adult/adult_data.csv

# lending
./xreason.py -p --pfiles lending.csv,lending ../bench/anchor/lending/
./xreason.py -c -t -n 50 ../bench/anchor/lending/lending_data.csv

# recidivism
./xreason.py -p --pfiles recidivism.csv,recidivism ../bench/anchor/recidivism/
./xreason.py -c -t -n 50 ../bench/anchor/recidivism/recidivism_data.csv

# compas
./xreason.py -p --pfiles compas.csv,compas ../bench/fairml/compas/
./xreason.py -c -t -n 50 ../bench/fairml/compas/compas_data.csv

# german
./xreason.py -p --pfiles german.csv,german ../bench/fairml/german/
./xreason.py -c -t -n 50 ../bench/fairml/german/german_data.csv
