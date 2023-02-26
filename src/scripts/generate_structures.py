#!/usr/bin/env python

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/my_code/src')

import argparse

import matplotlib.pyplot as plt

from ase.io import write
from nnp.struct_gen.struct_gen import generate_structures, plot_distribution

def main(
        fn_out: str,
        n_structures: int,
        min_density: float = 0.18914,
        max_density: float = 0.21756,
        max_carbs: int=60,
        min_carbs: int=10,
        h_stochiometry: float=4.,
        n_trials: int=100000,
        fn_plot: str=None
        ):
    
    structures = generate_structures(
        n_structures,
        min_density,
        max_density,
        max_carbs,
        min_carbs,
        h_stochiometry,
        n_trials
    )

    write(fn_out, structures)

    if fn_plot is None:
        exit()

    fig, ax = plt.subplots(1, figsize=(5,5))
    plot_distribution(structures, ax)
    plt.savefig(fn_plot)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a set of random structures'
    )

    parser.add_argument('fn_out', type=str, help='Output file name')
    parser.add_argument('-n', '--n_structures', type=int, required=True,
        help='Number of structures to generate')
    parser.add_argument('-d', '--min_density', type=float, required=False, default=0.18914,
        help='Minimum carbon density of the structure')
    parser.add_argument('-D', '--max_density', type=float, required=False, default=0.21756,
        help='Maximum carbon density of the structure')
    parser.add_argument('-c', '--max_carbs', type=int, required=False, default=60,
        help='Maximum number of carbons in the structure')
    parser.add_argument('-C', '--min_carbs', type=int, required=False, default=10,
        help='Minimum number of carbons in the structure')
    parser.add_argument('-H', '--h_stochiometry', type=float, required=False, default=4.,
        help='Number of hydrogens per carbon')
    parser.add_argument('-N', '--n_trials', type=int, required=False, default=100000,
        help='Number of trials to generate the structure')
    parser.add_argument('-p', '--fn_plot', type=str, required=False, default=None,
                        help='File name for the plot, if not provided no plot is generated')
    
    args = parser.parse_args()
    main(**vars(args))