#!/usr/bin/env python3

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read

import nnp
import argparse

def main(fn_traj, fn_image, cm_frame=False, time_step=.05):
    fig, ax = plt.subplots(1)
    
    plot_MSD(fn_traj, 'C', ax, cm_frame=cm_frame, time_step=time_step, label='C MSD')
    plot_MSD(fn_traj, 'H', ax, cm_frame=cm_frame, time_step=time_step, label='H MSD')
    
    ax.set_ylabel('MSD [$\AA^2$]')
    ax.set_xlabel('t [ps]')
    ax.legend()

    plt.savefig(fn_image, transparent=True)


def plot_MSD(
        fn_traj: str, 
        symbol: str, 
        ax: plt.axes, 
        time_step: float = .05,
        cm_frame: bool = False, 
        **kwargs) -> None:
    """Plots Mean Square distance on ax

    Parameters:
       fn_traj (str): file with trajectory stored 
       symbol (int): atom species to plot 
       ax: (plt.axes): where to plot
       time_step (float): time step 
       **kwargs: additional arguments for ax.plot()
    """

    traj = read(fn_traj, ':')
    n_steps = len(traj)
    
    # time dependetn MSD
    MSD = nnp.analysis.structure_descriptors._get_MSD(symbol, traj, cm_frame)
    time = np.linspace(0, n_steps, n_steps)*time_step
    
    ax.plot(time, MSD, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot MSD of a trajectory'
    )
    parser.add_argument('fn_traj', type=str, help='Trajectory file')
    parser.add_argument('fn_image', type=str, help='Output image file')
    parser.add_argument('-c', '--cm_frame', action='store_true', help='Center of mass frame')
    parser.add_argument('-t', '--time_step', type=float, default=.05, help='Time step in ps')

    args = parser.parse_args()
    main(args.fn_traj, args.fn_image, args.cm_frame, args.time_step)