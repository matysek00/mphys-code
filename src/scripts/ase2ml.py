#!/usr/bin/env python

"""Naive Script to convert any ase supported file to ml input
"""

import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse
from ase.io import read

from nnp.conversions.runner_convetor import Runner_Convertor
from nnp.conversions.ta_dah_convertor import Ta_dah_Convertor 

def convert(filname: str, outfil: str, tadah: bool = False, slice_at: str= ':', units: str = 'metalic'):
    
    traj = read(filname, slice_at)
    Conv = Ta_dah_Convertor if tadah else Runner_Convertor
    
    with open(outfil, 'w') as fil:
        writer = Conv(fil, units)
        writer.write_traj(traj)
        fil.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Naive Script to convert any ase supported file to ml input' 
    )

    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-t', '--tadah', action='store_true',
        help='output to a tadah geometry file')
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    parser.add_argument('-u','--units', type=str, required=False, default='metalic',
    help='In which units to store the data.')
    
    args = parser.parse_args()
    convert(args.infile, args.outfile, args.tadah, args.slice_at, args.units)
