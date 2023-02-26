#!/bin/bash

for fil in ../CH4-R3-AIMD-BLYP/*.castep
do
    outfil=${fil%.castep}
    ./castep-md2xyz.py $fil ../xyz-data/${outfil#CH4-R3-AIMD-BLYP/}.xyz
done
