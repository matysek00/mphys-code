#!/usr/bin/env python3

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import argparse
from ase.io import read
from nnp import md

def main(init_geo: str,
        dir_models: str,
        Temperature: float,
        n_steps: int = 10000,
        n_potentials: int = 4,
        exclude_models: list = [],
        logging_interval: int = 20,
        restart: bool = False):
    
    n_replicas = 1

    time_constant = 100 # fs
    time_step = .5 # fs

    cutoff = 4 # A
    shell= 1 # A
    fn_models = dir_models

    if n_potentials != 1:
        fmt_models = dir_models+'train-{:03d}/best_inference_model'
        fn_models = [fmt_models.format(i) for i in range(n_potentials) if i not in exclude_models]
    

    device = 'cpu'
    
    log_file = 'simulation.hdf5'
    
    if restart:
        log_file = 'simulation_restart.hdf5'

    chk_file = 'simulation.chk'
    buffer_size = 1 # how many steps to store in memory before writing to disk
    
    atoms = read(init_geo, '0')

    md_simulator = md.md.run_md_single(
        atoms = atoms,
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
        logging_interval=logging_interval,
        restart=restart
    )
    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('init_geo', type=str, 
                        help='Initial geometry')
    parser.add_argument('dir_models', type=str, 
                        help='Directory with models, or model filename if single model is used.')
    parser.add_argument('Temperature', type=float, 
                        help='Temperature')
    parser.add_argument('-n', '--n_steps', type=int, default=10000, 
                        help='Number of steps (default: 10000)')
    parser.add_argument('-m', '--n_models', type=int, default=4,
                        help='Number of models (default: 4)')
    parser.add_argument('-e', '--exclude_models', type=int, nargs='+', default=[],
                        help='Models to exclude (default: [])')
    parser.add_argument('-l', '--logging_interval', type=int, default=20,
                        help='Logging interval (default: 20)')
    parser.add_argument('-r', '--restart', action='store_true',
                        help='Restart simulation (default: False)')    
    
    args = parser.parse_args()
    main(args.init_geo, args.dir_models, args.Temperature, args.n_steps, args.n_models, args.exclude_models, args.logging_interval, args.restart)

