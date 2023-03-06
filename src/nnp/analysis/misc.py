import numpy as np
import ase

def get_mask(symbol: str, atoms: ase.Atoms) -> np.array:
    """np.array([at == symbol for at in atoms.get_chemical_symbols()])
    """
    mask = np.array([at == symbol for at in atoms.get_chemical_symbols()])
    return mask

def get_connectivity_matrix_wrapper(atoms: ase.Atoms) -> np.array:
    """Returns the connectivity matrix of the atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object.

    Returns
    -------
    np.array
        Connectivity matrix.
    """
    cutoff = ase.neighborlist.natural_cutoffs(atoms)
    nl = ase.neighborlist.NeighborList(
    cutoff, self_interaction=False, bothways=True)
    nl.update(atoms)
    connect_matrix = nl.get_connectivity_matrix()
    return connect_matrix