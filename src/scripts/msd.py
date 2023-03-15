#!/usr/bin/env python3

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read

import nnp
import argparse
from schnetpack.md.data import HDF5Loader


def main_hdf5(fn_traj, fn_output, cm_frame=False, conversion=10.):
    
    data = [HDF5Loader(fn) for fn in fn_traj]
    msd_C = nnp.analysis.structure_descriptors.get_msd(
        6, data, cm_frame=cm_frame, conversion=conversion)
    msd_H = nnp.analysis.structure_descriptors.get_msd(
        1, data, cm_frame=cm_frame, conversion=conversion)

    np.save('{}-MSD-H.npy'.format(fn_output), msd_H)
    np.save('{}-MSD-C.npy'.format(fn_output), msd_C)


def main_ase(fn_traj, fn_output, cm_frame=False):
        
        traj = []
        for fil in fn_traj:
            traj += read(fil, ':')
        
        msd_C = nnp.analysis.structure_descriptors.get_msd_atoms(
            'C', traj, cm_frame=cm_frame)
        msd_H = nnp.analysis.structure_descriptors.get_msd_atoms(
            'H', traj, cm_frame=cm_frame)
    
        np.save('{}-MSD-H.npy'.format(fn_output), msd_H)
        np.save('{}-MSD-C.npy'.format(fn_output), msd_C)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot MSD of a trajectory'
    )
    parser.add_argument('fn_data', type=str, nargs='+',
                        help='Trajectory files')
    parser.add_argument('-o', '--outfile', type=str, 
                        help='Output image file')
    parser.add_argument('-c', '--cm_frame', action='store_true', 
                        help='Center of mass frame')
    parser.add_argument('-k', '--conversion', default=10., type=float, 
                        help='Conversion to Angstroms')
    parser.add_argument('-d', '--hdf5', action='store_true',
                        help='Reading from HDF5 file')

    args = parser.parse_args()
    if args.hdf5:
        main_hdf5(args.fn_data, args.outfile, args.cm_frame, args.conversion)
    
    else:
        if args.conversion != 10.:
            print('Conversion only impemented with HDF5 files')
        main_ase(args.fn_data, args.outfile, args.cm_frame)