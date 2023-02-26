
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
            
    langevin = spkhooks.LangevinThermostat(Temperature, time_constant)
    COM_motion = spkhooks.RemoveCOMMotion(every_n_steps=1,
                                        remove_rotation=False, 
                                        wrap_positions=False)

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
    
    simulation_hooks = [
        langevin,
        file_logger,
        checkpoint,
        COM_motion
    ]
    
    return simulation_hooks


def get_calculator_old(
                cutoff: float,
                fn_models: list,
                device: str = 'cpu'
                ) -> SchNetPackEnsembleCalculator:
    """Create calculator for MD. 

    Parameters:
    cutoff (float): model cutoff
    fn_models (list): list of model paths 
    device (str): cpu or cuda

    Returns:
    md_calculator: Schentpack calculator
    """

    md_models = [torch.load(fn_mod, map_location=device).to(device) for fn_mod in fn_models]

    for model in md_models:
        model.requires_stress = False
        model.output_modules[0].stress = None

    md_calculator = SchNetPackEnsembleCalculator(
        md_models,
        required_properties=['energy', 'forces'],
        force_handle='forces',
        position_conversion='A',
        force_conversion='eV/A',
        neighbor_list=TorchNeighborList, 
        # ASENeighborList would be more appropriate but the 
        # memory keeps increasing during MD
        cutoff=cutoff
    )
    
    return md_calculator
    

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

    neighbor_list = NeighborListMD(
        cutoff,
        cutoff_shell,
        TorchNeighborList
    )

    if isinstance(fn_models, list):
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

    md_system.load_molecules(atoms, 
        n_replicas,
        position_unit_input="Angstrom"
        )
    
    velocities_set = atoms.get_velocities().any()

    if Temperature != None and not velocities_set:
        md_initializer = spk.md.initial_conditions.MaxwellBoltzmannInit(
            Temperature,
            remove_translation=True,
            remove_rotation=True)

        # Initialize momenta of the system
        md_initializer.initialize_system(md_system)
    
    return md_system