
import schnetpack as spk
import torch

from schnetpack.md.calculators import SchNetPackEnsembleCalculator, SchNetPackCalculator
from schnetpack.transform import  TorchNeighborList
from schnetpack.md.neighborlist_md import NeighborListMD
import schnetpack.md.simulation_hooks as spkhooks

from .paralle_ensemble_calculator import ParallelSchNetPackEnsembleCalculator

def get_hooks(
            log_file: str, 
            chk_file: str, 
            Temperature: float, 
            time_constant: float = .5,
            logging_interval: int = 100, 
            buffer_size: int = 10
            ) -> list:
    """Create hooks for the simulation
    
    Parameters:
    log_file (str): where to log 
    chk_file (str): where to store checkpoint
    Temperature (float): temperatur of the simulation
    time_constant (float): time constant for thermostat
    logging_interval(int): how often to log
    buffer_size (int): how long to store log in memory
    
    Returns: 
    simulation_hooks: list of shnetpack hooks
    """

    # Create the Langevin thermostat        
    langevin = spkhooks.LangevinThermostat(Temperature, time_constant)
    
    # remove center of mass motion
    com_motion = spkhooks.RemoveCOMMotion(every_n_steps=1,
                                        remove_rotation=False, 
                                        wrap_positions=False)

    # Create the file logger
    data_streams = [
        spkhooks.callback_hooks.MoleculeStream(store_velocities=True),
        spkhooks.callback_hooks.PropertyStream(),
    ]

    file_logger = spkhooks.callback_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps = logging_interval
    )

    # Create the checkpoint logger
    checkpoint = spkhooks.callback_hooks.Checkpoint(chk_file, every_n_steps=100)
    
    # Put it all together
    simulation_hooks = [
        langevin,
        file_logger,
        checkpoint,
        com_motion
    ]
    
    return simulation_hooks


    
def get_calculator(
                cutoff: float,
                fn_models: list,
                device: str = 'cpu'
                ) -> SchNetPackEnsembleCalculator:
    """Create calculator for MD. 

    Parameters:
    cutoff (float): model cutoff
    fn_models (list or str): list of model paths for ensemble or path to single model
    device (str): cpu or cuda

    Returns:
    md_calculator: Schentpack calculator
    """
    cutoff_shell = 2.0
    
    # create neighbor list
    neighbor_list = NeighborListMD(
        cutoff,
        cutoff_shell,
        TorchNeighborList
    )

    if isinstance(fn_models, list):
        # get ensemble calculator
        # TODO: try to parallelize the ensemble calculator
        md_calculator = SchNetPackEnsembleCalculator(
            fn_models,
            'forces',
            'eV',
            'Angstrom',
            neighbor_list,
            energy_key="energy",  # name of potential energies
            required_properties=[],
            script_model = False)
    
    else:
        # get single model calculator
        md_calculator = SchNetPackCalculator(
            fn_models,
            'forces',
            'eV',
            'Angstrom',
            neighbor_list,
            energy_key="energy",  # name of potential energies
            required_properties=[],
            script_model = False)
    
    return md_calculator

def get_system(
            atoms: list, 
            n_replicas: int = 1, 
            Temperature: float = None,
            device: str = 'cpu',
            ) -> spk.md.System:
    """Create schnetpack md system.

    Parameters:
    atoms (list): list of ase.atoms
    n_replicas (int): number of mds to be running under the same conditions.
    Temperature (float): temperatur of the simulation.
    device (str): cpu or cuda.

    Returns:
    md_system: Schentpack md system
    """

    md_system = spk.md.System()

    # load atoms into md system
    md_system.load_molecules(atoms, 
        n_replicas,
        position_unit_input="Angstrom"
        )
    
    # set velocities if they exist
    velocities_set = atoms.get_velocities().any()

    if Temperature != None and not velocities_set:
        # Initialize velocities
        md_initializer = spk.md.initial_conditions.MaxwellBoltzmannInit(
            Temperature,
            remove_translation=True,
            remove_rotation=True)

        md_initializer.initialize_system(md_system)
    
    return md_system