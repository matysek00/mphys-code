from typing import Union, List, Dict

import multiprocessing
import torch

from schnetpack.md.calculators.schnetpack_calculator import SchNetPackEnsembleCalculator

from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md import System

# TODO: DOES NOT WORK YET

class ParallelSchNetPackEnsembleCalculator(SchNetPackEnsembleCalculator):

    def __init__(
        self,
        model_files: List[str],
        force_key: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        neighbor_list: NeighborListMD,
        energy_key: str = None,
        stress_key: str = None,
        required_properties: List = [],
        property_conversion: Dict[str, Union[str, float]] = {},
        script_model: bool = True,):
        """
        Args:
            model_files (list(str)): List of paths to stored schnetpack model to be used in ensemble.
            force_key (str): String indicating the entry corresponding to the molecular forces
            energy_unit (float, float): Conversion factor converting the energies returned by the used model back to
                                         internal MD units.
            position_unit (float, float): Conversion factor for converting the system positions to the units required by
                                           the model.
            neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                        interatomic distances should be computed.
            energy_key (str, optional): If provided, label is used to store the energies returned by the model to the
                                          system.
            stress_key (str, optional): If provided, label is used to store the stress returned by the model to the
                                          system (required for constant pressure simulations).
            required_properties (list): List of properties to be computed by the calculator
            property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                               the model. Only changes the units used for logging the various outputs.
            script_model (bool): convert loaded model to torchscript.
        """

        super(ParallelSchNetPackEnsembleCalculator, self).__init__(
            model_files=model_files,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            stress_key=stress_key,
            required_properties=required_properties,
            property_conversion=property_conversion,
            script_model=script_model,
        )

        self.n_models = len(self.model)

    def _calcualte_single_model(self, idx_model,):
        """ self.model[idx_model](self.current_inputse)
        """
        model = self.model[idx_model]
        return model(self.current_inputs)
    
    def calculate(self, system: System):
        """
        Perform all calculations and compute properties and uncertainties.
        Args:
            system (schnetpack.md.System): System from the molecular dynamics simulation.
        """
        
        self.current_inputs = self._generate_input(system)
        print('alskjdflasjdfl')
        
        with torch.no_grad():
            with multiprocessing.Pool(self.n_models) as pool:
                results = pool.map(self._calcualte_single_model, range(self.n_models))      
        
        print('alskjdflasjdfl')
            
        # Compute statistics
        self.results = self._accumulate_results(results)
        self._update_system(system)








