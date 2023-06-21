#!/usr/bin/env python

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

from ase.io import read, write

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

def main(fil, fn_perm, outfile, distance, n_structures):
    
    name = fil.split('/')[-1]
    name = name.split('.')[0]
    fn_lam = fn_perm.replace('-perm', '-lam')

    perm = np.load(fn_perm)
    lam = np.load(fn_lam)[1:]

    if distance is not None:
        # selection based on distance    
        #we are ignoring the first element because first distance is always zero
        n_structures = np.where(lam<distance)[0][0]    
    
    idx_selected = perm[:n_structures].astype(int)

    print('{}: {:d} geometries selcted with {} maximum distance.'.format(
        name, n_structures, lam[n_structures]))

    data = read(fil, ':{:d}'.format(len(perm)))     
    data_selected = [data[idx] for idx in idx_selected]

    write(outfile, data_selected, append=True)

    #plot_selected_energies(data, idx_selected, name)


def plot_selected_energies(data, idx_selected, name='', timestep = .5):
    
    energies = np.array([x.get_potential_energy() for x in data])
    energies -= np.min(energies)
    
    n_steps = len(energies)
    time = timestep*np.linspace(0, n_steps-1, n_steps)
    
    matplotlib.rcParams.update({'font.size': 22})
    
    plt.figure(figsize=(30, 20))

    plt.plot(time, energies, c='black')
    plt.scatter(time[idx_selected], energies[idx_selected], c='red', label='Selected geometries', s=80)
    plt.scatter(time[idx_selected], np.zeros(len(idx_selected)), c='red', s=20)
    
    plt.legend()
    plt.title('Selected Geometries by FPS')
    plt.ylabel('Energy [eV]')
    plt.xlabel('Time [fs]')
    
    plt.savefig('FPS-energies-' + name + '.png', dpi=200)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(
        description='Select data after FPS was done.' 
    )
    parser.add_argument('infile', type=str, help='Original data')
    parser.add_argument('permfile', type=str, help='.npy file with permutations')
    parser.add_argument('outfile', type=str)
    parser.add_argument('-d', '--distance', type = float, required=False, default=None,
     help='What distance should be between point at the cuttof')
    parser.add_argument('-n', '--nstructures', type=int, required=False, default=None,
     help='How many structures to use. Either this or the distance parameter should be selected')


    args = parser.parse_args()
    selection_bad = ((args.distance is None and args.nstructures is None) or 
                    (args.distance is not None and args.nstructures is not None))
    
    if selection_bad:
        print('Please select either distance or the nstructures options, but not both')
        parser.print_help()
        exit()

    main(args.infile, args.permfile, args.outfile, args.distance, args.nstructures)
