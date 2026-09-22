# Sourced by measure.sh / gate_run.sh. Provides start_server / stop_server with a hard
# lifetime bound and an EXIT trap: the server can never outlive the measurement script,
# and the GPU lock (fd 9) is released only after the server is gone.
SERVER_PGID=""
SERVER_TIMEOUT=${SERVER_TIMEOUT:-2400}
stop_server() {
  [ -n "$SERVER_PGID" ] || return 0
  kill -INT -- -$SERVER_PGID 2>/dev/null
  for i in $(seq 1 60); do kill -0 -- -$SERVER_PGID 2>/dev/null || break; sleep 1; done
  kill -0 -- -$SERVER_PGID 2>/dev/null && { echo "SIGKILL server group $SERVER_PGID"; kill -9 -- -$SERVER_PGID 2>/dev/null; sleep 3; }
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$BIN"; then echo "killing leftover $p"; kill -9 $p; fi
  done
  for i in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 1; done
  SERVER_PGID=""
  echo "[$(date -u +%T)] server stopped"
}
trap 'stop_server; echo "[$(date -u +%T)] releasing lock"' EXIT
# start_server <logfile> [extra env...]  (uses $BIN $ENVF $PREFIX and argv.p35)
start_server() {
  local log=$1; shift
  setsid bash -c "cd $W/src && exec env -i HOME=$HOME PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
      \$(grep -v '^#' $ENVF | xargs -d '\n') $* timeout -s INT -k 60 $SERVER_TIMEOUT $PREFIX $BIN \$(cat $W/bench/argv.p35 | tr '\n' ' ')" > $log 2>&1 &
  SERVER_PGID=$!
  echo "server pgid $SERVER_PGID (lifetime bound ${SERVER_TIMEOUT}s)"
  for i in $(seq 1 900); do
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0
    kill -0 $SERVER_PGID 2>/dev/null || { echo "server died"; tail -30 $log; return 6; }
    sleep 1
  done
  echo "server never became healthy"; return 7
}
