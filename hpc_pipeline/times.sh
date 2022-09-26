#!/bin/bash
# collect time information from logfiles

echo "model,epoch,epoch-time,CPU-task,GPU-task,communication"
for log in $@; do
    sed -n -e 's/Model: \([0-9]*\) .* epoch \[\([0-9]*\)\/[0-9]*\], epoch time:\([^,]*\), CPU-task time:\([^,]*\), GPU-task time:\([^,]*\), CPU-GPU-communication time:\([^,]*\),.*/\1,\2,\3,\4,\5,\6/p' "$log"
done
