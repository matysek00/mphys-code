#!/usr/bin/env python

"""Naive Script to convert any ase supported file to ml input
"""

import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse
from ase.io import read, write

from nnp.analysis.edit_structure import multiply_trajectory

def convert(filnames: list, 
            outfil: str, 
            slice_at: str= ':', 
            interval: int = 1,
            append: bool = False,
            multiply: int = None):
    
    traj = []
    for filname in filnames:
        traj += read(filname, slice_at)[::interval]
    
    if multiply is not None:
        traj = multiply_trajectory(traj, multiply)
    
    write(outfil, traj, append=append)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Naive Script to rewrite geometries easily between two ase supported formats' 
    )

    parser.add_argument('infiles', type=str, nargs='+')
    parser.add_argument('-o', '--outfile', type=str, required=True)
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    parser.add_argument('-i', '--frame_interval', type = int, required=False, default=1,
                    help='Interval between frames to write out. Can\'t be done with slice_at.')
    parser.add_argument('-a','--append', action='store_true',
                        help='Append to the output file')
    parser.add_argument('-m','--multiply', type=int, required=False, default=None,
                        help='Multiply the trajectory by this number in each direction')
    
    args = parser.parse_args()
    convert(args.infiles, args.outfile, args.slice_at, args.frame_interval, args.append, args.multiply)
