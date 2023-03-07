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


def displace_positions(
                displacements: np.ndarray,
                positions: np.ndarray
                ) -> np.ndarray:
    """
    Displace positions of atoms in a geometry.
    
    Parameters
    ----------
    displacements : np.ndarray
        Displacements of atoms.
    positions : np.ndarray
        Positions of atoms.

    Returns
    -------
    new_positions : np.ndarray
        New positions of atoms.
    """

    n_atoms = len(positions)
    n_cells = len(displacements)

    # Apply displacements to positions
    new_positions = np.zeros((n_atoms*n_cells,3))
    for i in range(n_cells):
        new_positions[i*n_atoms:(i+1)*n_atoms,:] = positions + displacements[i,:]
    
    return new_positions


def get_cell_displacements(
        displacemt_amount: list = [2,2,2]
        ) -> np.ndarray:
    """Return displacements of atoms in a unit cell when multiplying geometry.

    Parameters
    ----------
    displacemt_amount : list, optional
        Number of times to multiply the geometry in each direction.
        The default is [2,2,2].

    Returns
    -------
    displacements : np.ndarray
        Displacements of atoms in a unit cell.
    """
    
    # Multiply unit cell by maximum increas in each direction
    m = max(displacemt_amount)
    indecies = list(range(m))

    # Create all possible combinations of displacements
    displacements_all = np.array(
        [item for item in itertools.product(indecies, repeat=3)])

    # Remove displacements that are not in the desired directions
    for i in range(3):
        displacements_all[:,i] = np.where(
            displacements_all[:,i] > displacemt_amount[i]-1, 
            0, displacements_all[:,i]) 
    
    displacements = np.unique(displacements_all, axis=0)
        
    return displacements