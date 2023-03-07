import numpy as np
import ase 

import itertools
from .misc import get_cell_displacements, displace_positions


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