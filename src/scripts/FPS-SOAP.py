#!/usr/bin/env python

import sys

sys.path.append('/storage/cmstore01/projects/Hydrocarbons/opt/my_code/src')

import argparse
import numpy as np
import matplotlib.pyplot as plt
import fps


def main(file, test_size):
    
    name = file.strip('npz')[:-1]
    
    DM = fps.get_DM(file, test_size) 
    perm, lam = fps.getGreedyPerm(DM)
    
    np.save(name+'-perm.npy', perm)
    np.save(name+'-lam.npy', lam)
    plot_FPS(lam, name)


def plot_FPS(lam, name):
    
    plt.plot(lam[1:])
    plt.yscale('log')
    
    plt.title('Furtherst Point Sampling')
    plt.ylabel('Furthest distance')
    plt.xlabel('Number of points')
    
    plt.savefig(name+'.png')


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(
        description='Perform Furthest Point Sampling (FPS) on SOAP descriptors.' 
    )

    parser.add_argument('infile', type=str, help='.npz file with with SOAP data.')
    parser.add_argument('-t', '--test_size', type=int, required=False, default=0,
        help='How many last structures to ignore.')
    
    args = parser.parse_args()

    main(args.infile, args.test_size)
