#!/bin/bash
# Run genome_train on a small tile to test the program is functioning
# correctly.
#

export MPLCONFIGDIR=/tmp/$USER-matplotlib

export CUDA_LAUNCH_BLOCKING=1
start=`date +%s`
complete() {
    end=`date +%s`
    echo "Completed in $((end-start)) seconds"
}
trap complete EXIT

rm -fr outputs
mkdir outputs
cd outputs
genome_train ../config.yaml ../tile.yaml

