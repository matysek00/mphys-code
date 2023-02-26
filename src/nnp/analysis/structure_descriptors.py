import numpy as np
from collections import Counter

import ase 
from ase.io import read
from ase.geometry.analysis import Analysis

from .misc import get_mask

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

def _get_MSD(symbol: str, traj: list, cm_frame: bool = False):
    
    # which atoms to use
    mask = get_mask(symbol, traj[0]) 
    
    # initial position
    postion0 = traj[0].get_positions()[mask]
    
    r_cm = np.zeros((len(traj), 3))
    
    if cm_frame:
        # initial center of mass
        r_cm0 = traj[0].get_center_of_mass()
        postion0 -= r_cm0

        # center of mass in time
        for i, atoms in enumerate(traj):
            r_cm[i] = atoms.get_center_of_mass()
        

    # get time dependetn MSD
    MSD = np.empty(len(traj))
    
    for i, atoms in enumerate(traj):
        MSD[i] = calculate_MSD(
            postion0,
            atoms.get_positions()[mask],
            r_cm[i])
    
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