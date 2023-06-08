#!/bin/sh

nwalkers=$1
nsteps=$2
nservers=$3

echo "python3 client.py -w ${nwalkers} -s ${nsteps} ${nservers}"
python3 client.py -w ${nwalkers} -s ${nsteps} ${nservers}
