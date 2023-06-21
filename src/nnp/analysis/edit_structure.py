import numpy as np
import ase 

from .misc import get_cell_displacements, displace_positions, get_mask


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
    if type(m) is int:
        m = m*np.ones(3, dtype=int)
        
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

def allign_hydgrogens(atoms):
    """Movest the images of H atoms so that they are closest to the C atoms
        I am using this to be able to manually remove entire molecules from 
        the system, without having to worry about missing H atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be modified

    Returns
    -------
    new_atom : ase.Atoms
        The atoms object with the H atoms moved
    """
    
    # just getting some useful information
    n_atoms = atoms.get_global_number_of_atoms()
    mask_H = get_mask('H', atoms)
    mask_C = get_mask('C', atoms)

    cell = atoms.get_cell()
    positions = atoms.get_positions()

    # the distance considering the minimum image convention
    distance_mic = atoms.get_all_distances(mic=True)[mask_H, :][:, mask_C]
    
    # the distance without considering the minimum image convention
    distance_no_mic = atoms.get_all_distances(mic=False)[mask_H, :][:, mask_C]

    # finding the H atoms whose current images are too far from the C atoms
    distance_ofset = distance_mic - distance_no_mic
    
    # finding the closest C atom to each H atom
    C_indx = np.argmin(distance_mic, axis=1)

    # the error on distance to closest C atom
    distance_ofset_from_molecule = distance_ofset[np.arange(len(C_indx)), C_indx]

    # if the error is too big, we need to change to a differnt image of the H atom
    idx_problematic_H = np.where(np.abs(distance_ofset_from_molecule) > 1e-6)[0]
    
    # only selecting the C atoms that are closest to the problematic H atoms
    idx_C = C_indx[idx_problematic_H]

    # converting the indices to be relative to the whole atoms object
    # not just the H or C atoms
    idx_problematic_H = np.arange(n_atoms)[mask_H][idx_problematic_H]
    idx_C = np.arange(n_atoms)[mask_C][idx_C]

    # getting the positions of the problematic H atoms and the closest C atoms
    positions_H = positions[idx_problematic_H]
    positions_C = positions[idx_C]

    # creating all possible images of the unit cell
    image_cells = get_cell_displacements([3,3,3]) - np.ones(3)
    image_cells = np.matmul(image_cells, cell)

    # creating all possible images of the H atoms
    image_positions = displace_positions(image_cells, positions_H).reshape(
        len(image_cells),len(positions_H), 3)

    # finding the closest image of the H atoms to the C atoms
    distance_images = np.linalg.norm(image_positions - positions_C[None,:, :], axis=-1)
    best_images = np.argmin(distance_images, axis=0)

    # choosing the displacements of the H atoms that 
    # are closest to the C atoms
    displacements = image_cells[best_images]

    # displacing the H atoms
    new_positions = positions
    new_positions[idx_problematic_H] += displacements

    # creating a new atoms object with the new positions
    new_atom = atoms.copy()
    new_atom.set_positions(newpositions=new_positions)

    return new_atom