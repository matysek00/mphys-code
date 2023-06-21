#!/usr/bin/env python

"""Naive Script to convert castep output file to any ase supported file.
"""

import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse
import numpy as np

from ase.io import read
from ase.geometry.analysis import Analysis

from nnp.analysis import edit_structure, structure_descriptors 

def main(infile: str,
        outfile: str =None,
        interval: int = 1, 
        slice_at: str = ':',
        nbins: int = 50,
        rmax: float = 11.255/2,
        multiply: bool = False,
        ) -> None:
    
    outfile = outfile +'_' if outfile is not None else ''

    traj = read(infile, slice_at)
    traj = edit_structure.multiply_trajectory(traj, m=multiply) if multiply is not None else traj

    analyser = Analysis(traj[::interval])

    save_rdf('C', 'C', nbins, rmax, analyser, outfile + 'RDF_CC.npy')
    save_rdf('C', 'H', nbins, rmax, analyser, outfile + 'RDF_CH.npy')
    save_rdf('H', 'H', nbins, rmax, analyser, outfile + 'RDF_HH.npy')


def save_rdf(at1: str, at2: str, nbins: int, rmax: float, analyser: Analysis, file: str):
    RDF = structure_descriptors.get_RDF(at1, at2, analyser, rmax, nbins)
    np.save(file, RDF)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write the RDF of a trajectory to a file.' 
    )

    parser.add_argument('infile', type=str,
                        help='The trajectory file to analyse')
    parser.add_argument('-n', '--nbins', type=int, default=100,
                        help='The number of bins to use in the RDF')
    parser.add_argument('-r', '--rmax', type=float, default=11.255/2,
                        help='The maximum distance to consider in the RDF')
    parser.add_argument('-i', '--interval', type=int, default=1,
                        help='The interval between points of the trajectory')    
    parser.add_argument('-o', '--output', type=str,
                        help='Output file name')
    parser.add_argument('-m', '--multiply', type=int, nargs='+', default=None,
                        help='Multiply uniti cell by this number in each direction, to get a longer range')
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    
    args = parser.parse_args()
    main(args.infile, args.output, args.interval, args.slice_at, args.nbins, args.rmax, args.multiply)
