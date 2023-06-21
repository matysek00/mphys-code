#!/usr/bin/env python3

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse

import numpy as np
import matplotlib.pyplot as plt

import pickle
import tables

from ase.io import iread

from scipy.stats import gaussian_kde
from nnp.analysis.testing import get_schnet_data, get_calculator, load_model_commitee


def color_hist(ax, size, x, y, line=False):
    """Plot a scatter plot with color based on point density."""
    if line:
        lim_max = max(np.max(x), np.max(y))
        lim_min = min(np.min(x), np.min(y))
        lim = (lim_max, lim_min)
        ax.plot(lim, lim, c='black', zorder=0)
    
    xy = np.vstack([x,y])

    # Calculate the point density
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, 
    # so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]    
    
    ax.scatter(x, y, c=z, s=size, zorder=1, cmap='viridis')


def test_model(fn_ref: str,
                dir_model: str,
                fn_out: str,
                fn_out_schnet: str = None,
                store_forces: bool = False,
                store_std: bool = False,
                single: bool = False,
                sample: int = None,
                ):
    """Test a model on a reference trajectory.

    Parameters:
        fn_ref (str): path to reference trajectory
        dir_model (str): path to model
        fn_out (str): path to output file for mean absolute error
        fn_out_schnet (str): path to output file for schnet data
        store_forces (bool): store forces in output file
        store_std (bool): store standard deviation in output file
        single (bool): test a single model
        sample (int): sample every n-th frame
    """

    init_files = False
    mae = {'mae_energy_atom': [], 'mae_forces_component': [], 'n_atoms': []}
    
    if single and store_std:
        print('Cannot store standard deviation for single model.')
        store_std = False
    
    if single:
        models = [get_calculator(dir_model)]
    else:
        models = load_model_commitee(dir_model)
    
    energies_schnet_std = None
    forces_schnet_std = None

    for i, atoms in enumerate(iread(fn_ref)):
        
        if i % sample != 0:
            # sampling does not work with in iread so I have to do it manually
            continue
        
        if fn_out_schnet is not None and not init_files:
            initilize_output_files(fn_out_schnet, store_forces, store_std, atoms)
            init_files = True
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        n_atoms = atoms.get_global_number_of_atoms()

        if store_std: 
            energies_schnet_mean, forces_schnet_mean, energies_schnet_std, forces_schnet_std = get_schnet_data([atoms], return_std=True, models = models)
        else:
            energies_schnet_mean, forces_schnet_mean = get_schnet_data([atoms], single_model=single, models=models)

        if fn_out_schnet is not None:
            store_schnet(fn_out_schnet, store_std, energies_schnet_mean, forces_schnet_mean, energies_schnet_std, forces_schnet_std)
        
        mae = update_mae(energy, forces, n_atoms, energies_schnet_mean, forces_schnet_mean, mae)

        del atoms, energies_schnet_mean, forces_schnet_mean
    
    with open(fn_out, 'wb') as f:
        pickle.dump(mae, f)


def store_schnet(fn_out_schnet, store_std, energies_schnet_mean, forces_schnet_mean, energies_schnet_std, forces_schnet_std):
    outfil = tables.open_file(fn_out_schnet, mode = 'a')

    outfil.root.forces.append(forces_schnet_mean)    
    outfil.root.energies.append(np.reshape(energies_schnet_mean, (1,1)))

    if store_std:
        outfil.root.forces_std.append(forces_schnet_std)
        outfil.root.energies_std.append(np.reshape(energies_schnet_std, (1,1)))
        del energies_schnet_std, forces_schnet_std
    outfil.close()
    

def initilize_output_files(fn_out, store_forces, store_std, atoms):
    """Initialize output files."""
    
    n_atoms = atoms.get_global_number_of_atoms()
    outfil = tables.open_file(fn_out, mode = 'w')
    atom = tables.Float32Atom()
            
    outfil.create_earray(outfil.root, 'energies', atom , shape=(0,1))
            
    if store_forces:
        outfil.create_earray(outfil.root, 'forces', atom, shape = (0, n_atoms, 3))
    
    if store_std:
        outfil.create_earray(outfil.root, 'energies_std',atom, shape = (0, 1))
        if store_forces:
            outfil.create_earray(outfil.root, 'forces_std', atom, shape = (0, n_atoms, 3))

    outfil.close()
                           

def update_mae(energy, forces, n_atoms, energies_schnet, forces_schnet, current_mae):
    """Update the mean absolute error of the model.

    Parameters:
        energy (float): reference energy
        forces (np.array): reference forces
        n_atoms (int): number of atoms in structure
        energies_schnet (np.array): predicted energies
        forces_schnet (np.array): predicted forces
        current_mae (dict): current mean absolute error
            current_mae['energy'] (list): list of mean absolute error of energies
            current_mae['forces'] (list): list of mean absolute error of forces
            current_mae['n_atoms'] (list): list of number of atoms in each structure

    Returns:
        current_mae (dict): updated mean absolute error
    """

    mae_energy = np.mean(np.abs(energies_schnet - energy)/n_atoms)
    mae_forces = np.mean(np.abs(forces_schnet - forces))
    
    current_mae['mae_energy_atom'].append(mae_energy)
    current_mae['mae_forces_component'].append(mae_forces)
    current_mae['n_atoms'].append(n_atoms)

    return current_mae    

def plot_energies_and_forces(ref_energies,
                            energies_schnet_mean,
                            ref_forces,
                            forces_schnet_mean,
                            n_atoms,
                            outfile):

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    color_hist(ax[0], 100, ref_energies/n_atoms, energies_schnet_mean/n_atoms)
    color_hist(ax[1], 10, ref_forces.flatten(), forces_schnet_mean.flatten(), line=True)

    ax[0].set_xlabel('Reference energy (eV/atom)')
    ax[0].set_ylabel('Schnet energy (eV/atom)')
    ax[1].set_xlabel('Reference force (eV/$\AA$)')
    ax[1].set_ylabel('Schnet force (eV/$\AA$)')

    plt.savefig('{}-plot.png'.format(outfile), dpi=100)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('fn_ref', type=str, 
                        help='Reference trajectories')
    parser.add_argument('-d', '--dir_model', type=str, 
                        help='Directory containing models')
    parser.add_argument('-o', '--outfile', type=str,
                        help='.pkl output file where mean absolute errors are stored')
    parser.add_argument('-s', '--outfile_schnet', type=str, default=None, 
                        help='.h5 file to store schnet predictions, if not given, only errors are stored')
    parser.add_argument('-f', '--store_forces', action='store_true',
                        help='Store forces in output file')
    parser.add_argument('-std','--store_std', action='store_true',
                        help='Calculate standard deviation')
    parser.add_argument('--single', action='store_true',
                        help='Only use single model')
    parser.add_argument('-n', '--sample', type=int, default=None,
                        help='Number of samples to use for each structure')
    
    args = parser.parse_args()

    test_model(args.fn_ref, args.dir_model, args.outfile, args.outfile_schnet, args.store_forces, args.store_std, args.single, args.sample)