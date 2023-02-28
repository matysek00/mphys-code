#!/usr/bin/env python

"""Calculate SOAP descriptors.
"""
import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys_code/src')

import argparse
from ase.io import read
import sparse

import fps

def main(infile: str, outfile: str, slice_at: str= ':'):

    data = read(infile, slice_at)
    soap_data = fps.calculate_soap(data)    

    sparse.save_npz(outfile, soap_data)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Calculate SOAP descriptors.' 
    )

    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    
    args = parser.parse_args()
    main(args.infile, args.outfile, args.slice_at)
