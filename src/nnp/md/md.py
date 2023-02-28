import os
import numpy as np

from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator

import schnetpack as spk
from schnetpack.md.data import HDF5Loader
from schnetpack import units as spk_units

from .initilize_md import get_system, get_calculator, get_hooks

_units ={'position': 'Angstrom',
        'energy': 'eV',
}
_conversions = {prop: 1/spk_units.unit2internal(unit) for prop, unit in _units.items()}

def run_md_single(atoms: list, 
                fn_models: list, 
                log_file: str,
                Temperature: float, 
                cutoff: float, 
                n_steps: int = 1000, 
                n_replicas: int = 1, 
                time_constant: float = 100, 
                time_step: float = .5, 
                device: str = 'cpu', 
                chk_file: str = 'simulation.chk', 
                buffer_size: int = 10, 
                logging_interval: int = 100, 
                ) -> spk.md.Simulator :
    """Run a MD simulation

    Parameters:
    ------------
    atoms : list
        initilal configuration of the system
    fn_models : list
        list of model files
    log_file : str
        where to log
    Temperature : float
        temperature of the simulation
    cutoff : float
        cutoff for the simulation
    n_steps : int
        number of steps to simulate
    n_replicas : int
        number of replicas to simulate
    time_constant : float
        time constant for the thermostat
    time_step : float
        time step for the simulation
    device : str
        device to run the simulation on
    chk_file : str
        where to store the checkpoint
    buffer_size : int
        how long to store the log in memory
    logging_interval : int
        how often to log

    Returns:
    --------
    md_simulator : spk.md.Simulator
        the simulator object
    """
    
    # print the input for reproducibility 
    print(locals())

    # initialize simulation
    md_system = get_system(atoms, n_replicas, Temperature, device)
    md_calculator = get_calculator(cutoff, fn_models, device)
    md_integrator = spk.md.integrators.VelocityVerlet(time_step)
    
    # createe thermostat and logs 
    simulation_hooks = get_hooks(
        log_file=log_file,
        chk_file=chk_file, 
        Temperature=Temperature,
        time_constant=time_constant,
        logging_interval=logging_interval, 
        buffer_size=buffer_size, 
        )
    
    # put it all together
    md_simulator = spk.md.Simulator(
            md_system, 
            md_integrator, 
            md_calculator, 
            simulator_hooks=simulation_hooks
        )
    
    # simulate
    md_simulator.simulate(n_steps)

    return md_simulator


def store_traj(fn_log: str,
            fn_traj: str,
            fn_ener: str, 
            fn_temp: str, 
            fn_force_var: str = None, 
            fn_energy_var: str = None,
            interval: int = 100) -> None:
    """Store the trajectory and the properties

    Parameters:
    -----------
    fn_log : str
        log file to read from
    fn_traj : str
        trajectory file to write to
    fn_ener : str
        whet tore store the energy
    fn_temp : str
        where to store the temperature
    fn_force_var : str
        where to store the force variance
    fn_energy_var : str
        where to store the energy variance
    interval : int
        how often to store the trajectory
    """

    
    # load data 
    traj, temperatures, energy, force_var, energy_var = load_data(
        fn_log, interval, fn_force_var is not None, fn_energy_var is not None)
    
    # store_data
    write(fn_traj, traj, append=True)
    store_var(temperatures, fn_temp, False)
    store_var(energy, fn_ener, False)
    
    if force_var is not None:
        store_var(force_var, fn_force_var, True)
    if energy_var is not None:
        store_var(energy_var, fn_energy_var, False)


def load_data(
        fn_log: str,
        interval: int,
        load_force_var: bool,
        load_energy_var: bool) -> tuple(list, np.array, np.array, np.array, np.array):
    """Load the data from the log file

    Parameters:
    -----------
    fn_log : str
        log file to read from
    interval : int
        interval between read structures
    load_force_var : bool
        whether to load the force variance
    load_energy_var : bool
        whether to load the energy variance

    Returns:
    --------
    traj : list
        trajectory
    temperatures : np.array
        temperature
    energy : np.array
        energy
    forces_var : np.array
        force variance
    energy_var : np.array   
        energy variance
    """

    data = HDF5Loader(fn_log)
    
    # rewrite_data
    traj = [hdf5_to_ase(data, i) for i in range(0,data.entries, interval)]
    temperatures = data.get_temperature()
    energy = data.get_property('energy', False)
    
    forces_var = None
    energy_var = None

    if load_force_var:
        forces_var = data.get_property('forces_var', True).sum(axis=-1)
    if load_energy_var:
        energy_var = data.get_property('energy_var', False)

    return traj, temperatures, energy, forces_var, energy_var


def store_var(new_var: np.array, fn_var: str, dim2: bool) -> None: 
    """Store the variance

    Parameters:
    -----------
    new_var : np.array
        variance to store
    fn_var : str
        where to store the variance
    dim2 : bool
        is the variance 2D
    """
    # reshape the data if they are 2D
    new_var_shaped = new_var
    if dim2:
        new_var_shaped = new_var.reshape(new_var.shape[0], -1)
    
    # add to the existing data if it exists
    var = new_var_shaped    
    if os.path.exists(fn_var):   
        var = np.concatenate([np.load(fn_var), new_var_shaped])

    # store the data
    np.save(fn_var, var)


def hdf5_to_ase(data, idx_structure: int) -> Atoms:
    """Convert the data from the hdf5 file to an ase Atoms object

    Parameters:
    -----------
    data : HDF5Loader
        data to convert
    idx_structure : int
        index of the structure to convert

    Returns:
    --------
    atoms : Atoms
        the converted structure
    """
    
    species = data.get_property('_atomic_numbers', True)
    cell = data.get_property('_cell', False)[0]*_conversions['position']
    positions = data.get_property('_positions', True)[idx_structure]*_conversions['position']
    velocities  = data.get_property('velocities', True)[idx_structure]*_conversions['position']
    energy  = data.get_property('energy', False)[idx_structure]*_conversions['energy']
    forces  = data.get_property('forces', True)[idx_structure]*_conversions['energy']/_conversions['position']
    atoms = Atoms(symbols = species, positions=positions, cell=cell, velocities = velocities)
    atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy, forces=forces)
    atoms.set_pbc((True, True, True))
    
    return atoms