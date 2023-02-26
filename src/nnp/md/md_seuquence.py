import os
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry.analysis import Analysis

from schnetpack.md.data import HDF5Loader

from itertools import compress

from .md import store_traj, hdf5_to_ase, run_md_single

distance_pairs = [
    ('C', 'C', 1.0), 
    ('C', 'H', .75), 
    ('H', 'H', .5)
]


def iterative_md(init_geo: str,
                fn_models: list, 
                Temperature: int,
                nmax: int = 100, # maximum number of iterations
                n_steps: int = 1000, # steps between extrapolation check is done
                n_replicas: int = 1, # number of MDs each different startin conditios
                time_constant: float = 100, # fs
                time_step: float = .5, # fs  
                cutoff: float = 4.0, # A 
                buffer_size: int = 10, # how many geometries to hold in  memory before writing them down
                logging_interval: int = 1, # how often to store data into the temporary log file
                store_interval: int = 100, # how often to store geometires into fn_store
                tresh_store: float = .9, # geometries with average forcre std per atom above this value will counted as extrapolation
                tresh_stop: float = 1,   # if std above this value is reached the MD will stop
                fn_store: str = 'traj.db', # where to store intermidiate trajectory
                fn_extrapol: str = 'extrapol.db', # where to store extrapolation values 
                device: str = 'cpu', 
                chk_file: str = 'simulation.chk', # where to store checkpoints
                fn_ener: str = 'energy.npy', # where to store energies
                fn_energy_var: str = 'var_energy.npy', # where to store energy variance
                fn_force_var: str = 'var_force.npy', # where to store force variance
                fn_temp: str = 'temperature.npy', # where to store temperature
                remove_logs: bool = True,
                factor_blow_up = .9 # end the simulation when distance between two atoms gets below this value times distance pair
                ) -> None:
    
    # print the input for reproducibility 
    print(locals())
    
    # load in the initial geometry
    atoms = read(init_geo, '-1')
    
    for i in range(nmax): 
        log_file= 'simulation-{:d}.hdf5'.format(i)

        # run md for n_steps    
        md_simulator = run_md_single(
            atoms=atoms, 
            fn_models=fn_models, 
            log_file=log_file,
            Temperature=Temperature, 
            cutoff=cutoff, 
            n_steps=n_steps, 
            n_replicas=n_replicas, 
            time_constant=time_constant, 
            time_step=time_step, 
            device=device, 
            chk_file=chk_file, 
            buffer_size=buffer_size, 
            logging_interval=logging_interval 
        )

        # get the final state of atoms
        atoms = md_simulator.system.get_ase_atoms()[0]
        
        # store data from the simulation and figure whether to run further
        max_force_std, end = finalize_md(
            fn_store=fn_store,
            log_file=log_file,
            fn_ener=fn_ener,
            fn_force_var=fn_force_var,
            fn_energy_var=fn_energy_var,
            fn_temp=fn_temp,
            interval=store_interval,
            remove_logs=remove_logs,
            fn_extrapol=fn_extrapol,
            tresh_store=tresh_store,
            factor_blow_up=factor_blow_up
        )
    
        print('!!! Iteration : {:d}, max force component std: {:.2f} eV/A'.format(
            i, max_force_std))
            

        if max_force_std > tresh_stop:
            print('!!! Hit extrapolations - Simulation will terminate')
            break
        
        if end: 
            print('Atoms too close - Simulations will terminate.')
            break    
        
    print('Simulation Done')


def finalize_md(fn_store: str, 
            log_file: str,
            fn_ener: str = 'energy.npy', 
            fn_force_var: str = 'var_forece.npy', 
            fn_energy_var: str = 'var_energy.npy',
            fn_temp: str = 'temperature.npy', 
            fn_extrapol: str = 'extrapol.db', 
            interval: int = 100,
            tresh_store: float = .9,
            remove_logs: bool = True,
            factor_blow_up = .9
            ) -> tuple:

    # store all the data into different files
    store_traj(fn_log=log_file, 
            fn_traj=fn_store, 
            fn_ener=fn_ener, 
            fn_force_var=fn_force_var, 
            fn_energy_var=fn_energy_var,
            fn_temp=fn_temp, 
            interval=interval 
    )
    
    # figure out to the largest extratpolation 
    data = HDF5Loader(log_file)
    max_force_std = np.max(force_std(data))

    # store extrapolative geometries and check if the atoms don't come top cloes 
    end = write_extrapol(fn_extrapol, log_file, tresh_store, factor_blow_up)
    
    if remove_logs:
        os.remove(log_file)

    return max_force_std, end


def write_extrapol(fn_extrapol: str,
                log_file: str, 
                tresh_store: float, 
                factor: float = .9
                ) -> bool:
    
    # load data 
    data = HDF5Loader(log_file)
    
    # check where forces are above the treshold
    idxs_structure = np.where(force_std(data) > tresh_store)[0]

    # create geometries
    traj = [hdf5_to_ase(data, idx) for idx in idxs_structure]

    # check for too short distances
    not_blown_up = [not check_blow_ups(t, 1) for t in traj]
    print('!! Found {:d} extrapolations, {:d} are useful'.format(len(idxs_structure), sum(not_blown_up)))

    # not blown up 
    traj_good = list(compress(traj, not_blown_up)) 
    write(fn_extrapol, traj_good, append=True)

    # shouls we stop the simulation
    complete_blow = bool(sum([check_blow_ups(t, factor) for t in traj]))

    return complete_blow


def force_std(data, tradeoff: float = 8.1) -> float:

    force_average_std = np.sqrt(data.get_property('forces_var', True).sum(axis=2).mean(axis=1))
    force_max_std = np.sqrt(data.get_property('forces_var', True).max(axis=2).mean(axis=1))
    
    return force_average_std + force_max_std/tradeoff


def check_blow_ups(geometry: Atoms, factor: float = 1) -> bool:

    global distance_pairs
    ana = Analysis(geometry)
    blow_up = 0
    
    for p in distance_pairs:
        # find bonds and their lengths
        bonds = ana.get_bonds(p[0], p[1], unique=True)
        
        # check that bonds actually exist
        if len(bonds[0]) == 0:
            continue

        bond_lengths = ana.get_values(bonds)
        
        # check if the minumum length is below the treshhold value
        blow_up += np.min(bond_lengths) < factor*p[2]
        
    return bool(blow_up)