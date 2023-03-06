import numpy as np
import matplotlib.pyplot as plt

import ase
from scipy import sparse

from collections import Counter
from .misc import get_connectivity_matrix_wrapper

def get_molecules(atoms: ase.Atom) -> Counter:
    """Count molecules in the structure.
    
    Parameters:
    ----------
      atoms (ase.Atom): structure of atoms
    
    Returns:
    -------
        n_components (counter): counter of molecules
    """

    # get conectivity matix
    connect_matrix = get_connectivity_matrix_wrapper(atoms)

    # remove hydrogen-hydrogen bonds from the connectivity matrix
    # otherwise methanes will be merged into one molecule
    indx_H = np.where(atoms.numbers == 1)[0]
    mask = np.ones(connect_matrix.shape, dtype=bool)
    # TODO: geneailize this so it works even if hydrogens are not at the begining
    mask[:indx_H[-1], :indx_H[-1]] = False
    
    # apply mask
    connect_matrix_masked = np.multiply(connect_matrix.todense(), mask)
    connect_matrix = sparse.dok_matrix(connect_matrix_masked)

    n_components, component_list = sparse.csgraph.connected_components(
    connect_matrix)

    # turn into individula molecules with componenets
    idx_molecules = []
    for i_mol in range(n_components):
        mol = np.where(component_list==i_mol)[0]
        idx_molecules.append(mol)

    # get symbols for each molecule
    symbols = atoms.get_chemical_symbols()
    symbol_molecules = []
    for mol in idx_molecules:
        count = Counter([symbols[i] for i in mol])
        string_mol = ''
        if 'C' in count: 
           string_mol+='C{:}'.format(count['C'])
        if 'H' in count: 
           string_mol+='H{:}'.format(count['H'])
        
        # remove 1s from symbols
        string_mol = string_mol.replace('C1H', 'CH')
        string_mol = string_mol.replace('H1', 'H')
        symbol_molecules.append(string_mol)

    count_molecules = Counter(symbol_molecules)
    return count_molecules


def get_unique_keys(arr: list):
    """Find unique keys in a list of dictioneries.
    
    Parameters:
    ----------
        arr (list): list of dictionaries
    
    Returns:
    -------
        unique (list): list of unique keys
    """
    
    unique = []
    for d in arr:
        unique.extend(list(d))
    unique=list(set(unique))
    return unique


def mol_evol(molecules: list):
    """Return unique molecules with their count in the structu in time

    Parameters:
    ----------
        molecules (list): list of dictionaries with molecular symbol an its count

    Returns:
    -------
        time_molecules (dict): 
            dictionary with unique molecules as keys and their count as a function of time
    """
    
    # get unique molecules in the entire trajectory
    unique_mols = get_unique_keys(molecules)
    
    # get their count as a funciton of time
    time_molecules = {}
    n_steps = len(molecules)

    # Calculate number of molecules in time
    for mol in unique_mols:
        counts = np.empty(n_steps, dtype=int)
        for i, step in enumerate(molecules):
            counts[i] = step[mol] if mol in step else 0
        time_molecules[mol] = np.array(counts)

    return time_molecules