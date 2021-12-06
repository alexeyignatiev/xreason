#!/bin/sh

cd ../

# adult
cut -d ',' -f 1-12 bench/anchor/adult/adult_data.csv > data
tail -n +2 data | sort -u > bench/anchor/adult/adult.samples

# lending
cut -d ',' -f 1-9 bench/anchor/lending/lending_data.csv > data
tail -n +2 data | sort -u > bench/anchor/lending/lending.samples

# recidivism
cut -d ',' -f 1-15 bench/anchor/recidivism/recidivism_data.csv > data
tail -n +2 data | sort -u > bench/anchor/recidivism/recidivism.samples

# compas
cut -d ',' -f 1-11 bench/fairml/compas/compas_data.csv > data
tail -n +2 data | sort -u > bench/fairml/compas/compas.samples

# german
cut -d ',' -f 1-21 bench/fairml/german/german_data.csv > data
tail -n +2 data | sort -u > bench/fairml/german/german.samples

rm data
