import numpy as np
import matplotlib.pyplot as plt

import ase
from ase import Atoms
from ase.ga import startgenerator

# Define the minimum bond length for the structure generator
_blmin = {(6, 6): 1.1, (1,1): 0.6, (1,6): .85, (6,1): .85}

def random_structure(a: float,
                    n_carbs: int,
                    h_stochiometry: float=4.,
                    n_trials: int=100000) -> Atoms:
    """Generate random structure
    
    Parameters: 
    -----------
    a: float
        Cell size
    n_carbs: int
        Number of carbons in the structure
    h_stochiometry: float
        Number of hydrogens per carbon
    n_trials: int
        Number of trials to generate the structure
    
    Returns:
    --------
    canditate: ase.Atoms
        Generated structure
    """
    
    # Round the number of hydrogens to the nearest integer
    n_hydro = int(round(h_stochiometry*n_carbs))
    
    # Prepare the generator
    slab = Atoms(pbc=[True, True,True], cell=[a,a,a])
    blocks = [('C', n_carbs), ('H', n_hydro)]
    generator = startgenerator.StartGenerator(
        slab,
        blocks, 
        _blmin, 
        test_too_far=False
    )
    
    # Generate the structure
    canditate = generator.get_new_candidate(n_trials)
    
    return canditate


def generate_structures(n_structures: int,
                        min_density: float = 0.18914,
                        max_density: float = 0.21756,
                        max_carbs: int=60,
                        min_carbs: int=10,
                        h_stochiometry: float=4.,
                        n_trials: int=100000
                        ) -> list:
    """Generate random structures
    
    Parameters:
    -----------
    n_structures: int
        Number of structures to generate
    min_density: float
        Minimum carbon density of the structure
    max_density: float
        Maximum carbon density of the structure
    max_carbs: int  
        Maximum number of carbons in the structure
    min_carbs: int
        Minimum number of carbons in the structure
    h_stochiometry: float
        Number of hydrogens per carbon
    n_trials: int  
        Number of trials to generate the structure
        
    Returns:
    --------
    structures: list
        List of ASE Atoms objects
    """
    
    structures = []
    i = 0

    while i < n_structures:
        
        # Generate random number of atoms and cell size
        n_carbs, a = random_input(
            min_density, 
            max_density, 
            max_carbs, 
            min_carbs
            )
        
        # Generate random structure
        candidate = random_structure(
            a, 
            n_carbs, 
            h_stochiometry=h_stochiometry, 
            n_trials=n_trials
        )

        if candidate is None:
            continue

        structures.append(candidate)
        i+=1
        
    return structures


def random_input(min_density: float,
                max_density: float,
                max_carbs: int,
                min_carbs: int
                ):
    """Generate random number of atoms and cell size
    for the structure generator.
    
    Parameters:
    -----------
    min_density: float
        Minimum carbon density of the structure
    max_density: float
        Maximum carbon density of the structure
    max_carbs: int
        Maximum number of carbons in the structure
    min_carbs: int
        Minimum number of carbons in the structure
    
    Returns:
    --------
    n_carbs: int
        Number of carbons in the structure
    a: float
        Cell size
    """
    
    # Generate random number of atoms and density
    n_carbs = np.random.randint(min_carbs, max_carbs)
    density = np.random.uniform(min_density, max_density)
    
    # Calculate the cell size
    a = (n_carbs/density)**(1/3)
    
    return n_carbs, a

def plot_distribution(structures: list, 
                      ax: plt.axes
                    ) -> None:
    """Plot the distribution of the number of 
    atoms and the density of the structures.

    Parameters
    ----------
    structures : list
        List of structures.
    ax : plt.axes
        Axes to plot on.
    """
    sizes = np.array([geo.get_number_of_atoms() for geo in structures])
    volumes = np.array([geo.get_volume() for geo in structures])
    densities = sizes/volumes

    ax.scatter(sizes, densities)
    ax.set_xlabel('Number of atoms')
    ax.set_ylabel('Density (atoms/Å³)')