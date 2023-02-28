#!/usr/bin/env python

"""Calculate SOAP descriptors.
"""
import sys
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys_code/src')

import argparse
from ase.io import read, write

def main(infile: str, outfile: str, slice_at: str= ':', interval: int = 1, template_file: str = None):
    fmt_output = '{}-{:04d}.cell'

    if template_file is not None:
        with open(template_file, 'r') as f:
            template = f.read()

    traj = read(infile, slice_at)[::interval]
    
    for i, atoms in enumerate(traj):
        output = fmt_output.format(outfile, i)
        
        if template_file is not None:
            with open(output, 'w') as f:
                f.write(template)

        write(output, atoms, append=(template_file is not None))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Convert an ASE trajectory to CASTEP cell files.' 
    )

    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str,
                        help='The outputs will be written into <outfile>-idx.cell')
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    parser.add_argument('-i', '--frame_interval', type = int, required=False, default=1,
                        help='Interval between frames to write out. Can\'t be used with slice_at.')
    parser.add_argument('-t', '--template_file', type=str, required=False, default=None,
                        help='Template file to use for the cell file. (Default: None)')
    
    args = parser.parse_args()
    main(args.infile, args.outfile, args.slice_at, args.frame_interval, args.template_file)
