import random
import numpy as np
from ase.units import create_units

# TODO: A better way to do this would be to implement this in ase

class General_Convertor():
    """General class to be inherited by convertors.
    Parameters: 
        file: file objected opende for reading or writing/appendig
        units (str): units to be used in the file (metalic or atomic)
    """
    unit_sets = {
        'metalic': {'energy': 'eV', 'position': 'Ang'},
        'atomic': {'energy': 'Hartree', 'position': 'Bohr'}
        } 

    def __init__(self, file, units: str = 'metalic'):
        self.file = file
        self.units = units
        
        # set the conversion factors
        conversion = create_units('2014')
        self.conv_ener = conversion['eV']/conversion[self.unit_sets[units]['energy']]
        self.conv_pos = conversion['Ang']/conversion[self.unit_sets[units]['position']]
        self.conv_force = self.conv_ener/self.conv_pos        

    def read_structure(self):
        return NotImplementedError

    def write_traj(self, traj: list, n: int = None) -> list:
        """
        Write a sample of trajectory into a file 
        Parameters: 
            traj (list): list of atoms object
            n (int): how many to write into the file if n is None all will be used
        Returns 
            new_traj (list): list of unused structres
        """

        # sample trajectory (for splinting into training and test sets)
        sample = traj if n is None else random.sample(traj, n)
        
        for frame in sample: 
            self.write(frame)
        
        # return the unused structures
        new_traj = [x for x in traj if x not in sample]
        return new_traj

    def write_energy(self, frame):
        """Write the energy of the frame into the file.
        """

        if frame.get_calculator() is not None:
            energy = frame.get_potential_energy()*self.conv_ener
        else: 
            energy = 0.0
            
        self.file.write(self.fmt_energy.format(energy))
    
    def write_atom(self, frame, names):
        """Write the atoms of the frame into the file.
        """

        if frame.get_calculator() is not None:
            forces = frame.get_forces()
        else: 
            forces = np.zeros((len(names), 3))

        for i, name in enumerate(names):
            self.fill_atom(frame, forces, i, name) 

    def read_matrix(self, n: int = 3):
        """read a matrix with n lines
        """
        matrix =[]
        
        for _ in range(n):
            line = self.file.readline().split()
            matrix.append([float(x) for x in line])
        
        return np.array(matrix)