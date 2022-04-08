#!/bin/bash
# with_postgresql.sh

TILE=$1
shift;

host=`hostname` # e.g. batch1
dbname=optuna_training
password=mackerel_fish
port=5654
export STUDY="$TILE"
export PGSQL_URL="postgresql://optuna:$password@$host:$port/$dbname"
# This line can be read by programs watching this script:
echo "Running $STUDY $PGSQL_URL"

# consider using /mnt/bb/$USER/pgdata
export PGDATA=/gpfs/alpine/$PROJ/scratch/$USER/pgdata

if [ ! -d $PGDATA ]; then
  # Directory does not exist, create here:
  mkdir -p $PGDATA

  initdb
  pg_ctl -o "-p $port -h $host" -l logfile start
  sleep 0.1

  createdb -p $port $dbname

  psql -p $port $dbname <<.
create user optuna with encrypted password '$password';
grant all privileges on database $dbname to optuna;
.

  cat >>$PGDATA/pg_hba.conf <<.
host    all             all             samenet                 md5
.
  pg_ctl reload

else
  # directory exists, just launch the pg
  pg_ctl -o "-p $port -h $host" -l logfile start
fi

optuna create-study --skip-if-exists --study-name "$STUDY" --storage "$PGSQL_URL"

# run program here:
echo "Running: $@"
$@

pg_ctl stop
# if you put the database somewhere temporary, delete it:
# rm -fr /mnt/bb/$USER
