# tear down the redis server

file=redis.pid
if [ ! -s "$file" ]; then
  echo "Unable to read pid file $file."
  exit 1
fi
pid=`tail -n1 "$file"`
kill -15 $pid
rm "$file"
