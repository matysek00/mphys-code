import numpy as np
import ase

def get_mask(symbol: str, atoms: ase.Atoms) -> np.array:
    """np.array([at == symbol for at in atoms.get_chemical_symbols()])
    """
    mask = np.array([at == symbol for at in atoms.get_chemical_symbols()])
    return mask