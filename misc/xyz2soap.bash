#!/bin/bash

for fil in ../xyz-data/*.xyz
do
    outfil=${fil%.xyz}
    ./xyz2soap.py $fil ../SOAP-data/${outfil#../xyz-data/}-SOAP.npz
done
