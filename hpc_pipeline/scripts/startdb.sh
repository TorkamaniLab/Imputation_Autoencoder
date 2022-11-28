#!/bin/bash
# setup a redis server locally
# we assume this is only done by 1 process per node
# (or else you would need a unique PORT per process).

set -e
DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

host=`hostname` # e.g. batch1
password=`dd if=/dev/urandom count=1 2> /dev/null | sha256sum | cut -c-20`
port=$((6000 + RANDOM%4000))
export STUDY=genomeai
export STORAGE="redis://default:$password@$host:$port"

sed -e "s/PASSWORD/$password/g; s/PORT/$port/g;" "$DIR/redis.conf" >redis.conf

# This line can be read by programs watching this script:
# It must be the first line output!
echo "Running $STUDY $STORAGE"

redis-server redis.conf >redis.log &
pid=$!
disown $pid
echo $pid >redis.pid

optuna create-study --direction maximize --skip-if-exists \
                    --study-name "$STUDY" --storage "$STORAGE"
