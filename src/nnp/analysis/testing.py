import schnetpack as spk
import schnetpack.transform as trn
import torch
import numpy as np
import matplotlib.pyplot as plt

import ase
from ase.io import read

def get_calculator(fn_model: str, cutoff: float = 4.0) -> spk.interfaces.SpkCalculator:
    """Get calculator for a given model.
    
    Parameters:
        fn_model (str): path to model
        cutoff (float): cutoff for the calculator

    Returns:
        calculator (spk.interfaces.SpkCalculator): calculator
    """

    calulator = spk.interfaces.SpkCalculator(
        model_file = fn_model,
        neighbor_list=trn.ASENeighborList(cutoff=cutoff),
        energy_key = 'energy',
        forces_key = 'forces',
        energy_unit ='eV',
        position_unit='Ang'
    )
    
    return calulator


def get_schent_energy(
        atoms: ase.Atoms, 
        calculator: spk.interfaces.SpkCalculator
        ) -> float:
    """Get energy of a structure using SchNetPack.

    Parameters:
        atoms (ase.Atoms): structure
        calculator (spk.interfaces.SpkCalculator): calculator

    Returns:
        energy (float): energy of the structure
    """
    atoms.set_calculator(calculator)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    return energy, forces

def get_schnet_energies(atoms: ase.Atoms,
    calculators: list
    )-> np.array :
    """Get energies of a structure using SchNetPack.

    Parameters:
        atoms (ase.Atoms): structure
        calculators (list): list of calculators
    
    Returns:
        energies (np.array): energies of the structure
    """
    energies = np.zeros(len(calculators))
    forces = np.zeros((len(calculators), len(atoms), 3))
    for i, calculator in enumerate(calculators):
        energies[i], forces[i] = get_schent_energy(atoms, calculator) 
    
    return energies, forces


def get_schnet_data(
        ref_traj: list, 
        return_std: bool = False, 
        single_model: bool = False,
        dir_model: str = None,
        models: spk.interfaces.SpkCalculator = None):    
    
    if models is None:
        if single_model:
            models = [get_calculator(dir_model)]
        else:
            models = load_model_commitee(dir_model)
    
    # Get energies and forces from schnet
    data_schnet  = [get_schnet_energies(atoms, models) for atoms in ref_traj] 
    energies_schnet = np.array(
        [data_schnet[i][0] for i in range(len(data_schnet))])
    forces_schnet = np.array(
        [data_schnet[i][1] for i in range(len(data_schnet))])   
    
    # average over models
    energies_schnet_mean = np.mean(energies_schnet, axis=1) 
    forces_schnet_mean = np.mean(forces_schnet, axis=1)

    if return_std:
        energies_schnet_std = np.std(energies_schnet, axis=1) 
        forces_schnet_std = np.std(forces_schnet, axis=1)
        return energies_schnet_mean, forces_schnet_mean, energies_schnet_std, forces_schnet_std

    return energies_schnet_mean,forces_schnet_mean

def load_model_commitee(dir_model):
    fmt_model = dir_model + 'train-{:03d}/best_inference_model'
    fn_models = [fmt_model.format(i) for i in range(4)]
    models = [get_calculator(fn_mod) for fn_mod in fn_models]
    return models

def seperation_energy(
    fn_data: str,
    dir_model: str,
    fn_reference: str = None,
    align: bool = True, 
    n_models: int = 4,
    exclude: list = [],
    ) -> tuple((np.array, np.array, np.array)):
    """Calculate seperation energy.

    Parameters:
        fn_data (str): path to data
        fn_reference (str): path to reference data
        dir_model (str): path to models
        align (bool): use offset to align energies
        n_models (int): number of models
        exclude (list): list of model indecies to exclude

    Returns:
        mean_schnet (np.array): mean of energies
        std_schnet (np.array): std of energies
        if fn_reference is not None:
            reference_energies (np.array): energies from reference
    """
    
    # read data
    traj = read(fn_data, index=':')
    
    # get models
    fmt_model = dir_model+'/train-{:03d}/best_inference_model'
    fn_models = [fmt_model.format(i) for i in range(n_models) if i not in exclude]

    # get energies
    calculators = [get_calculator(mod) for mod in fn_models]
    schnet_energies = np.array([get_schnet_energies(atoms, calculators) for atoms in traj])

    # calculate mean and std of schnet energies
    mean_schnet = np.mean(schnet_energies, axis=1)
    std_schnet = np.std(schnet_energies, axis=1)

    if fn_reference is not None:
        # read reference data
        traj_reference = read(fn_reference, index=':')
        reference_energies = np.array([atoms.get_potential_energy() for atoms in traj_reference])

        # align energies
        offset = np.min(reference_energies) - np.min(mean_schnet) if align else 0
        mean_schnet += offset
        schnet_energies += offset

        return mean_schnet, std_schnet, reference_energies
    
    return mean_schnet, std_schnet