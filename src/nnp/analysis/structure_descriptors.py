import numpy as np
from collections import Counter

from scipy.spatial import distance_matrix

import ase 
from ase.io import read
from ase.geometry.analysis import Analysis

from .misc import get_mask, get_connectivity_matrix_wrapper

def calculate_MSD(x1: np.array, 
                  x2: np.array, 
                  r_cm: np.array = np.zeros(3)
                  )-> float:
    """3*np.mean((x1 - (x2 - r_cm))**2)
    """

    # factor of 3 is there because I am averaging over the 
    # 3 position components while I want to sum over them
    MSD = 3*np.mean((x1 - (x2 - r_cm))**2)
    return MSD

def get_msd_atoms(mask: str,
                  traj: list, 
                  cm_frame: bool = False):
    """Returns MSD

    Parameters
    ----------
    mask : str, or array
        mask for the atoms to be used, if int, then it is the atomic number
    traj : list
        trajectory
    cm_frame : bool, optional
        if True, then the center of mass is subtracted from the positions, by default False

    Returns
    -------
    MSD : np.array
        time dependent MSD
    """
    
    # which atoms to use
    if isinstance(mask, str):
        mask = get_mask(mask, traj[0])

    r_0 = traj[0].get_positions()[mask]
        
    all_postitions = np.array([atoms.get_positions()[mask] for atoms in traj])
    if cm_frame:
        r_cm = np.array([atoms.get_center_of_mass() for atoms in traj])
        all_postitions -= r_cm[:, None, :]

    msd = 3*np.mean((all_postitions - r_0)**2, axis=(1,2))
    return msd


def get_msd(mask: int,
             data: list, 
             cm_frame: bool = False, 
             conversion: float = 1.):
    """Returns MSD

    Parameters
    ----------
    mask : int, or array
        mask for the atoms to be used, if int, then it is the atomic number
    data : list
        list of hdf5 dataloaders
    cm_frame : bool, optional
        if True, then the center of mass is subtracted from the positions, by default False
    conversion : float, optional
        conversion factor for the positions, by default 1.

    Returns
    -------
    MSD : np.array
        time dependent MSD
    """
    
    # which atoms to use
    if isinstance(mask, int):
        mask = data[0].get_property('_atomic_numbers', True) == mask
    
    # number of steps
    n_steps = [0] + [dat.entries for dat in data]
    n_steps = np.cumsum(n_steps)
    total_steps = n_steps[-1]

    # get positions from all data files
    all_positions = np.empty((total_steps, mask.sum(), 3))

    for i in range(len(data)):
        start = n_steps[i]
        end = n_steps[i+1]

        all_positions[start:end] = data[i].get_positions()[:, mask, :]*conversion

    # initial position
    postion0 = all_positions[0]

    r_cm = 0
    r_cm0 = 0
    
    if cm_frame:
        # initial center of mass
        r_cm0 = np.mean(postion0, axis=0)
        postion0 -= r_cm0

        # center of mass in time
        r_cm = np.mean(all_positions, axis=1)
        all_positions -= r_cm[:, None, :]
        
    # get time dependetn MSD
    MSD = 3*np.mean((all_positions - postion0)**2,axis=(1,2))
    
    return MSD

def get_RDF(at1:str, 
            at2:str,
            analyser: Analysis,
            rmax: float = 11.255/2, 
            nbins: int = 100, 
            scale: int = 1
            )-> np.array:
    """Returns RDF

    Parameters
    ----------
    at1 : str
        atomic symbol of center atom
    at2 : str
        atomic symbol of second atom
    analyser : Analysis
        ASE analyser object
    rmax : float, optional
        maximum value of r, by default 11.255/2
    nbins : int, optional
        number of bins, by default 100
    scale : int, optional
        scale factor for the RDF, by default 1

    Returns
    -------
    RDF : np.array
        Radial distribution function
    """

    rdfs = np.array(analyser.get_rdf(rmax, nbins, elements=[at1, at2]))
    RDF = np.average(rdfs, axis=0)*scale
    return RDF

def calculate_CN(
    central_atom: str,
    secondary_atom: str, 
    atoms: ase.Atoms) -> float:

    """Calculate coordianation number.

    Parameters: 
       central_atom (str): central element
       secondary_atom (str): secondary element
       atoms (ase.Atoms): structure on which it is calculated
    """
    
    # how many central atoms are there
    n_atoms = np.sum(get_mask(central_atom, atoms))

    ana = Analysis(atoms)
    bonds = ana.get_bonds(central_atom, secondary_atom)
    
    # dictionary where key is atom 
    # and value is its coordianation number
    c = Counter([b[0] for b in bonds[0]])

    # average coordination number 
    coordiantion_number = np.sum(list(c.values()))/n_atoms
    
    return coordiantion_number


def cn_in_time(fn_traj: str, 
            atom_pairs: list)-> np.array:
   
   """Plot average coordination number as a function of time.

   Parameters: 
      fn_traj (str): file with trajectory stored
      atom_pairs (list): [[central_atom1, secondary_atom1], 
                        [central_atom2, secondary_atom2],...]
            
   Returns:
      all_coordination_numbers (np.array): 
         coordination numbers for all atom pairs in the trajectory
   """
    
   traj = read(fn_traj, ':')
   n_steps = len(traj)
    
   all_coordination_numbers = np.empty((len(atom_pairs), n_steps))
   for i, pair in enumerate(atom_pairs):
      # get time dependent coordiantaion number
      Coord_Num = np.array([calculate_CN(pair[0],pair[1], atoms) for atoms in traj])
      all_coordination_numbers[i,:] = Coord_Num

   return all_coordination_numbers

def structure_distance_matrix(atoms: ase.Atoms, 
                              sym: tuple = None, 
                              triangular: bool = True) -> np.array:
    """Calculate distance matrix for a given structure.

    Parameters
    ----------
    atoms : ase.Atoms
        structure
    sym : tuple, optional
        tuple of atomic symbols to use, by default None (all atoms)
        eg ('C', 'C') will retrurn only CC distances
        ('C', 'H') will return only CH distances etc.
    triangular : bool, optional
        if True, only lower triangle of the matrix will be non infinite   
    
    Returns
    -------
    DM : np.array
        distance matrix
    """

    posittions = atoms.positions
    
    mask_1 = np.ones(len(atoms), dtype=bool)
    mask_2 = np.ones(len(atoms), dtype=bool)

    if sym is not None:
        # if sym is not None, only use atoms with given symbols
        mask_1 = get_mask(sym[0], atoms)
        mask_2 = get_mask(sym[1], atoms)

    # calculate distance matrix
    connect_matrix = get_connectivity_matrix_wrapper(atoms)
    DM = get_distance_matrix(connect_matrix)
    
    if triangular:
        # set redundant part of the matrix to zero, 
        # only loweer triangle will remain nonzero
        DM = np.tril(DM, k=-1)

    # replace zeros with inf
    DM = np.where(DM == 0, np.inf, DM)
    
    return DM