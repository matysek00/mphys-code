#!/usr/bin/env python3

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

from scipy.stats import gaussian_kde
from nnp.analysis.testing import get_schnet_energies, get_calculator

def color_hist(ax, size, x, y, line=False):
    
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

def test_model(fn_ref, dir_model, outfile):

    # Load models
    fmt_model = dir_model + 'train-{:03d}/best_inference_model'
    fn_models = [fmt_model.format(i) for i in range(4)]
    models = [get_calculator(fn_mod) for fn_mod in fn_models]

    # Load reference data
    ref_traj = []
    for fil in fn_ref:
        ref_traj += read(fil, ':')
    
    # get number of atoms
    n_atoms = np.array(
        [atoms.get_global_number_of_atoms() for atoms in ref_traj])

    # Get energies and forces from schnet
    data_schnet  = [get_schnet_energies(atoms, models) for atoms in ref_traj] 
    energies_schnet = np.array(
        [data_schnet[i][0] for i in range(len(data_schnet))])
    forces_schnet = np.array(
        [data_schnet[i][1] for i in range(len(data_schnet))])   
    
    # average over models
    energies_schnet_mean = np.mean(energies_schnet, axis=1) 
    forces_schnet_mean = np.mean(forces_schnet, axis=1)

    # get reference energies and forces
    ref_energies = np.array(
        [atoms.get_potential_energy() for atoms in ref_traj])
    ref_forces = np.array(
        [atoms.get_forces() for atoms in ref_traj])
    
    # Plot
    fig, ax = plt.subplots(1,2, figsize=(10,5))

    color_hist(ax[0], 100, ref_energies/n_atoms, energies_schnet_mean/n_atoms)
    color_hist(ax[1], 10, ref_forces.flatten(), forces_schnet_mean.flatten(), line=True)

    ax[0].set_xlabel('Reference energy (eV/atom)')
    ax[0].set_ylabel('Schnet energy (eV/atom)')
    ax[1].set_xlabel('Reference force (eV/$\AA$)')
    ax[1].set_ylabel('Schnet force (eV/$\AA$)')

    plt.savefig(outfile, dpi=300)

    # Calculate MAE
    mae_energy = np.mean(np.abs(energies_schnet_mean - ref_energies)/n_atoms)
    mae_forces = np.mean(np.abs(forces_schnet_mean - ref_forces)/n_atoms[:,None,None])

    print(r'MAE energy: {:.3f} meV/atom'.format(mae_energy*1000))
    print(r'MAE forces: {:.3f} meV/$\AA$/atom'.format(mae_forces*1000))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('fn_ref', type=str, nargs='+', 
                        help='Reference trajectories')
    parser.add_argument('-d', '--dir_model', type=str, 
                        help='Directory containing models')
    parser.add_argument('-o', '--outfile', type=str, 
                        help='Where to save the plot')
    
    args = parser.parse_args()

    test_model(args.fn_ref, args.dir_model, args.outfile)