#!/usr/bin/env python3

import sys 
sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/mphys-code/src')

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read

import nnp
import argparse
from schnetpack.md.data import HDF5Loader


def main(fn_traj, fn_image, cm_frame=False, time_step=.05, conversion=10.):
    fig, ax = plt.subplots(1)
    
    plot_MSD(fn_traj, 6, ax, cm_frame=cm_frame, time_step=time_step, conversion=conversion, label='C MSD')
    plot_MSD(fn_traj, 1, ax, cm_frame=cm_frame, time_step=time_step, conversion=conversion,label='H MSD')
    
    ax.set_ylabel('MSD [$\AA^2$]')
    ax.set_xlabel('t [ps]')
    ax.legend()

    plt.savefig(fn_image, transparent=True)


def plot_MSD(
        fn_data: list, 
        symbol: str, 
        ax: plt.axes, 
        time_step: float = .05,
        cm_frame: bool = False, 
        conversion: float = 10.,
        **kwargs) -> None:
    """Plots Mean Square distance on ax

    Parameters:
       fn_traj (str): file with trajectory stored 
       symbol (int): atom species to plot 
       ax: (plt.axes): where to plot
       time_step (float): time step 
       **kwargs: additional arguments for ax.plot()
    """

    n_steps = [0]
    MSD = []

    # load data
    for fn in fn_data:
        data = HDF5Loader(fn)
        n_steps.append(data.entries)
        # time dependetn MSD
        MSD.append(nnp.analysis.structure_descriptors.get_MSD(symbol, data, cm_frame, conversion))

    total_steps = np.sum(n_steps)
    MSD_tot = np.empty(total_steps)

    for i in range(len(MSD)):
        MSD_tot[n_steps[i]: n_steps[i+1]] = MSD[i]

    time = np.linspace(0, total_steps, total_steps)*time_step
    
    ax.plot(time, MSD, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot MSD of a trajectory'
    )
    parser.add_argument('fn_data', type=str, nargs='+', help='Trajectory files')
    parser.add_argument('-o', '--outfile', type=str, help='Output image file')
    parser.add_argument('-c', '--cm_frame', action='store_true', help='Center of mass frame')
    parser.add_argument('-k', '--conversion', default=10., type=float, 
                        help='Conversion to Angstroms')
    parser.add_argument('-t', '--time_step', type=float, default=.05, help='Time step in ps')

    args = parser.parse_args()
    main(args.fn_data, args.outfile, args.cm_frame, args.time_step, args.conversion)