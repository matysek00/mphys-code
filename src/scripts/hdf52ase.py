#!/usr/bin/env python

"""Naive Script to convert any ase supported file to ml input
"""

import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse
from ase.io import read, write

from schnetpack.md.data import HDF5Loader
from nnp.analysis.edit_structure import multiply_trajectory

def convert(filnames: list, 
            outfil: str, 
            skip_initial: int = 0, 
            interval: int = 1,
            append: bool = False,
            multiply: int = None):
    
    traj = []

    for filname in filnames:
        data = HDF5Loader(filname, skip_initial=skip_initial, load_properties=False)
        traj += data.convert_to_atoms()[::interval]
    
    if multiply is not None:
        traj = multiply_trajectory(traj, multiply)
    
    write(outfil, traj, append=append)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Script to rewrite Schnetpack MD data to ase readable format.\
            Does not include energy or forces.' )

    parser.add_argument('infiles', type=str, nargs='+')
    parser.add_argument('-o', '--outfile', type=str, required=True)
    parser.add_argument('-s', '--skip_initial', type = int, required=False, default=0,
        help='Skip the first n frames.')
    parser.add_argument('-i', '--frame_interval', type = int, required=False, default=1,
                    help='Interval between frames to write out. Can\'t be done with slice_at.')
    parser.add_argument('-a','--append', action='store_true',
                        help='Append to the output file instead of overwriting it.')
    parser.add_argument('-m','--multiply', type=int, required=False, default=None,
                        help='Multiply the trajectory by this number in each direction')
    
    args = parser.parse_args()
    convert(args.infiles, args.outfile, args.skip_initial, args.frame_interval, args.append, args.multiply)
