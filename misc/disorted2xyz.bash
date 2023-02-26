#!/bin/bash

for fil in ../CH4-Mol-Distortions/*.castep
do
    
    outfil=${fil%.castep}
    outfile=../disorted_xyz/${outfil#../CH4-Mol-Distortions/}.xyz
    
    if grep -q "Checkpoint file cannot be written." $fil
    then 
        echo $fil Failed DFT
    else
        ./castep-scf2xyz.py $fil $outfile
    fi
done

for fil in ../disorted_xyz/*.xyz
do
    cat $fil >> ../xyz-data/CH4-Mol-Distortions.xyz
done

