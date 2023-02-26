from .general_convertor import General_Convertor


class Runner_Convertor(General_Convertor):
    """Class for writting n2p2 trajectory files.
    Parameters: 
        file: file objected opened for writing/appendig
    """

    def __init__(self, file, units: str = 'metalic'):
        super(Runner_Convertor, self).__init__(file, units)
        
        self.fmt_one = '{:13.6f}'
        self.fmt_atom = 'atom ' + 3*self.fmt_one + '{:^6s}' + 2*self.fmt_one.format(0.0) + 3*self.fmt_one + '\n'
        self.fmt_energy = 'energy ' + self.fmt_one + '\n'

    def read_structure(self):
        return super().read_structure()
    
    def write(self, frame):
        """Write a single geometry into a file
        """
    
        # get a list of names
        names = [at.symbol for at in frame]

        fmt_lattice = 'lattice ' + 3*self.fmt_one + '\n'   
        fmt_charge = 'charge ' + self.fmt_one + '\n'

        self.file.write('begin\n')

        if frame.info is not None:
            self.file.write('comment ' + str(*frame.info) + '\n')

        if frame.cell is not None:
            for lattice_vector in frame.cell:
                self.file.write(fmt_lattice.format(*lattice_vector*self.conv_pos))

        self.write_atom(frame, names)
        self.write_energy(frame)

        self.file.write(fmt_charge.format(0.0))
        self.file.write('end\n')
    
    def fill_atom(self, frame, forces, i, name):
        self.file.write(self.fmt_atom.format(*frame.positions[i]*self.conv_pos,
                    name,
                    *forces[i]*self.conv_force))