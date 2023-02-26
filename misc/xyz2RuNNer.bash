#!/bin/bash

for fil in ../xyz-data/*.xyz
do
    outfil=${fil%.xyz}
    ./xyz2RuNNer.py $fil ../RuNNer-data/${outfil#../xyz-data/}.data
done
