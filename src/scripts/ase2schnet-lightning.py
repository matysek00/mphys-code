#!/usr/bin/env python

"""Naive Script to convert any ase supported file to schnetpack input
"""
import argparse
from ase.io import read
import numpy as np
import schnetpack as spk


def convert(fn_data: list, outfil: str, slice_at: str= ':'):
    
    # load trajectory
    traj = []
    for fil in fn_data:
        traj += read(fil, slice_at)

    new_dataset = spk.data.create_dataset(
        datapath=outfil,
        format=spk.data.AtomsDataFormat.ASE,
        distance_unit='Ang',
        property_unit_dict=dict(
            energy='eV', 
            forces='eV/Ang',
        ),
    )

    # add data to dataset
    property_list = []
    for struct in traj:
        energy = np.array([struct.get_potential_energy()], dtype=np.float32)
        forces = np.array(struct.get_forces(),dtype= np.float32)
        
        property_list.append(dict(
            energy=np.array(energy),
            forces=np.array(forces),
        ))


    # add data to dataset
    new_dataset.add_systems(property_list=property_list, atoms_list=traj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Naive Script to convert any ase supported file to schnetpack input' 
    )

    parser.add_argument('infile', type=str, nargs='+')
    parser.add_argument('outfile', type=str)
    parser.add_argument('-s', '--slice_at', type = str, required=False, default=':',
        help='What range of structure to use, use python slicing')
    
    args = parser.parse_args()
    convert(args.infile, args.outfile, args.slice_at)
