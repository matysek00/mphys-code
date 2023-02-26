#!/usr/bin/env python

"""Naive Script to convert castep output file to any ase supported file.
"""

import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/my_code/src')

import argparse
from ase.io import write
from nnp.conversions.castep_convertor import Castep_SCF_Convertor, Castep_MD_Convertor


def convert(files: list, 
            outfil: str, scf: 
            bool = False, 
            pbc: bool = True,
            finite_set_correction: bool = True):
    
    Conv = Castep_SCF_Convertor if scf else Castep_MD_Convertor 
    traj = []

    for filname in files:
        with open(filname, 'r') as file:
            reader = Conv(file, finite_set_correction=finite_set_correction)
            traj += reader.read(pbc=pbc)
    write(outfil, traj, append=True)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Naive Script to convert castep output file to any ase supported file.' 
    )

    parser.add_argument('infiles', type=str, nargs='+')
    parser.add_argument('-o', '--outfile', type=str, default='output.db')
    parser.add_argument('-s', '--scf', action='store_true',
        help='Is the input file a restult of and scf calculation')
    parser.add_argument('-v', '--vacuum', action='store_true',
        help='Don\'t use pbc')
    parser.add_argument('-f', '--finite_set_correction', action='store_true',
        help='Use if no finite set correction was used in the calculation') 
    
    
    args = parser.parse_args()
    convert(args.infiles, args.outfile, args.scf, (not args.vacuum), (not args.finite_set_correction))
