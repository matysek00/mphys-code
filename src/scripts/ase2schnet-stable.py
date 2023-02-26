#!/usr/bin/env python

"""Naive Script to convert any ase supported file to schnetpack input
"""

import argparse
from ase.io import read
import numpy as np
import schnetpack as spk


def convert(filname: str, outfil: str, slice_at: str= ':', properties: bool = True):
    
    # load trajectory
    traj = read(filname, slice_at)

    # read properties
    property_list= []
    available_properties = []
    if properties:
        property_list = get_property_list(traj)
        available_properties=['energy', 'forces']

    # initiliaze dataset
    new_dataset = spk.AtomsData(outfil, 
        available_properties=available_properties)
    
    # add data to dataset
    new_dataset.add_systems(traj, property_list)


def get_property_list(traj):
    property_list = []
    for struct in traj:
        energy = np.array([struct.get_potential_energy()], dtype=np.float32)
        forces = np.array([struct.get_forces()], dtype= np.float32)
        
        property_list.append(
            {'energy': energy,
            'forces': forces}
        )
        
    return property_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Naive Script to convert any ase supported file to schnetpack input' 
    )

    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    parser.add_argument('-p', '--properties', action='store_true', default=False,
        help='Use when energies and forces are not available.')
    
    args = parser.parse_args()
    convert(args.infile, args.outfile, args.slice_at, not args.properties)
