# Mphys Code 

This code was used for my masters project, on modeling high-preassure, high-temperature hydrocarbons. Most of the code are wrappers to [schnetpack](https://schnetpack.readthedocs.io/en/latest/) functions and [ase](https://wiki.fysik.dtu.dk/ase/index.html).

The code is split into two parts since some libraries seemed incompatible (probably solvable but it was easier to split it). 

## NNP

The main part of the code concerns the trainining anp aplication of neural network potentials (NNPs), specifically schnet. There is not much interesting stuff here, these are mainly wrappers so I can keep reusing the same settings.
