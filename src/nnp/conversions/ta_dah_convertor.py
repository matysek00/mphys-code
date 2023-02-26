import ase
from ase.calculators.singlepoint import SinglePointCalculator

from .general_convertor import General_Convertor


class Ta_dah_Convertor(General_Convertor):
    """Class for reading and writting ta-dah trajectory files.
    Parameters: 
        file: file objected opende for reading or writing/appendig
        weights (list): weights for model predictions (see ta-dah documentation)
                        if None they won't be used.   
    """

    def __init__(self, file, weights: list = None, units: str = 'metalic'):
        super(Ta_dah_Convertor, self).__init__(file, units)
        
        self.fmt_one = '{:.13f} '
        self.fmt_atom = '{:s} ' + 6*self.fmt_one + '\n'
        self.fmt_energy = self.fmt_one + '\n'
        
        
        self.weights = weights
        self.use_weights = self.weights is None

    def read(self):
        traj = []
        file_end = False

        while not file_end:

            try:
                cell, positions, forces, symbols, energy,stress = self.read_structure()
        
                atoms = ase.Atoms(symbols = symbols, positions=positions, cell=cell)
                atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy, forces=forces, stress=stress)
                traj.append(atoms)
            
            except:
                file_end = True

        return traj

    def read_structure(self):

        positions = []
        forces = []
        symbols = []

        info = self.file.readline()
        if self.use_weights:
            self.weights = [float(x) for x in self.file.readline().split()]

        energy = float(self.file.readline())

        cell = self.read_matrix()
        stress = self.read_matrix()

        line = self.file.readline().split()

        while len(line) != 0:
            symbols.append(line[0])
            positions.append([float(x) for x in line[1:4]])
            forces.append([float(x) for x in line[4:7]])
            line = self.file.readline().split()
        
        return cell,positions,forces,symbols,energy,stress
    
    def write(self, frame):
        """Write a single geometry into a file
        """

        fmt_lattice = 3*self.fmt_one + '\n'
        fmt_weights = 3*self.fmt_one + '\n'

        names = [at.symbol for at in frame]

        if frame.info is not None:
            self.file.write('Comment: ' + str(*frame.info) + '\n')
        else:
            self.file.write('No comment\n')
        
        if self.use_weights:
            self.file.write(fmt_weights.format(*self.weights))

        self.write_energy(frame)

        if frame.cell is not None:
            for lattice_vector in frame.cell:
                self.file.write(fmt_lattice.format(*lattice_vector*self.conv_pos))
        else:
            for _ in range(3):
                self.file.write(fmt_lattice.format(0,0,0))

        try:
            for stress_vector in frame.get_stresses():
                self.file.write(fmt_lattice.format(stress_vector*self.conv_ener/self.conv_pos**3))
        except:
            for _ in range(3):
                self.file.write(fmt_lattice.format(0,0,0))

        self.write_atom(frame, names)

        self.file.write('\n')
    
    def fill_atom(self, frame, forces, i, name):
        self.file.write(self.fmt_atom.format(name, 
                    *frame.positions[i]*self.conv_pos,
                    *forces[i]*self.conv_force))  
                    