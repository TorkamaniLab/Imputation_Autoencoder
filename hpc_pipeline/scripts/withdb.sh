#!/bin/bash
# Launch a command after setting up a redis server on RANK 0
# This script is called in parallel, from every rank.

if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
   startdb.sh >startdb.log
   if [ $? -eq 0 ]; then
     info=`head -n1 startdb.log`
   else
     info=Error
   fi
   trap "stopdb.sh" EXIT
fi

# broadcast the server info (on rank 0) to all MPI ranks
info=(`broadcast $info`)
if [[ x"${info[0]}" != x"Running" ]]; then
   exit 1
fi
export STUDY=${info[1]}
export STORAGE=${info[2]}

echo "Command: $@"
$@
