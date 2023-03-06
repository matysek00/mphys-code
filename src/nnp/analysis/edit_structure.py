import numpy as np
import ase 

import itertools

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


def multiply_geometry(geometry: ase.Atoms, m: int = 2):
    """
    Multiply geometry by 2 in each direction.

    Parameters
    ----------
    geo : ase.Atoms
        Geometry of atoms.
    m : int or list, optional
        Number of times to multiply the geometry. The default is 2.

    Returns
    -------
    new_geo : ase.Atoms
        New geometry of atoms, with a bigger unit cell and more atoms.
    """

    # make m a list
    if m is int:
        m = m*np.ones(3)
    
    # Create new unit cell
    cell = geometry.cell
    new_cell = m*cell

    # Create displacements of atoms to fill new cell
    # relative to the original cell
    displacements_cell = get_cell_displacements(m)

    # Generate displacements in Angstroms
    displacements = np.matmul(displacements_cell, cell)

    # Displace positions
    new_positions = displace_positions(displacements, geometry.positions)

    # Create new symbols
    n_cells = len(displacements)
    new_symbols = geometry.get_chemical_symbols()*n_cells

    # Create new geometry
    new_geo = ase.Atoms(
        symbols=new_symbols, positions=new_positions, 
        cell=new_cell, pbc=geometry.pbc)
    
    return new_geo




def merge_geometries(geos: tuple, direction: int = 2):
    """
    Merge geometries into one geometry.

    Parameters
    ----------
    geos : tuple
        two ase.Atoms objects.

    Returns
    -------
    new_geo : ase.Atoms
        New geometry of atoms, 
        where the original are stuck on top of each other.
    """
    displacement_cell = np.zeros((1,3))
    displacement_cell[:,direction] = 1

    displacements = np.matmul(displacement_cell, geos[0].cell)
    
    new_positions = np.vstack(
        [geos[0].positions,
        displace_positions(displacements, geos[1].positions)])

    new_symbols = geos[0].get_chemical_symbols() + geos[1].get_chemical_symbols()
    new_cell = geos[0].cell
    new_cell[direction] += geos[1].cell[direction]

    new_geo = ase.Atoms(
        symbols=new_symbols, positions=new_positions, 
        cell=new_cell, pbc=geos[0].pbc
    )

    return new_geo


def multiply_trajectory(traj: list, m: int = 2):
    """
    Multiply trajectory by 2 in each direction.

    Parameters
    ----------
    traj : list
        List of ase.Atoms objects.
    m : int, optional
        Number of times to multiply the geometry. The default is 2.

    Returns
    -------
    new_traj : list
        New list of ase.Atoms objects, with 2x2x2 unit cell.
    """
    new_traj = []
    for geo in traj:
        new_traj.append(multiply_geometry(geo, m=m))
    return new_traj