
from schnetpack.md import System
from schnetpack.md.integrators import VelocityVerlet
import numpy as np

from schnetpack.md.simulation_hooks.basic_hooks import SimulationHook
from schnetpack.md.simulator import Simulator


class ConstrainedVelocityVerlet(VelocityVerlet):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.
    Adde a very naive way to fix positions of some atoms. 
    It is quite inefficient, as the potential will still calculate all forces.

    Args:
        time_step (float): Integration time step in femto seconds.
        mask (np.ndarray): Mask for the atoms to be propagated.
    """

    ring_polymer = False
    pressure_control = False

    def __init__(self, time_step: float, mask: np.ndarray):
        super(ConstrainedVelocityVerlet, self).__init__(time_step)
        self.mask = mask


    def half_step(self, system: System):
        """
        Half steps propagating the system momenta according to:
        ..math::
            p = p + \frac{1}{2} F \delta t
        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """

        system.momenta[:,self.mask, :] = system.momenta[:,self.mask, :] + 0.5 * system.forces[:,self.mask, :] * self.time_step



class ConstrainedMotion(SimulationHook):
    """
    Hook to fix the motion of some atoms after thermostat.
    Still need to remove forces from the atoms, with an integrator.
    Alwasy include as the last hook.

    Args:
        mask (np.ndarray): Mask for the atoms to be constrained.
    """

    def __init__(self, mask: np.ndarray):
        super(ConstrainedMotion, self).__init__()
        self.mask = mask

    def _remove_motion(self, simulator: Simulator):
        """
        simulator.system.momenta[:,self.mask, :] = 0

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        simulator.system.momenta[:,self.mask, :] = 0

    def on_step_begin(self, simulator: Simulator):
        self._remove_motion(simulator)
        

    def on_step_end(self, simulator: Simulator):
        self._remove_motion(simulator)
        


