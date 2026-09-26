#!/bin/bash
# Wait for the sealed archive, download, verify the hash, then destroy the instance immediately.
# Destruction is requested ONLY after the archive hash matches the remote receipt.
SSH="ssh -i $HOME/.ssh/vastai.priv -p 50675 -o ConnectTimeout=15 root@71.17.164.141"
SCP="scp -q -i $HOME/.ssh/vastai.priv -P 50675"
OUT=/Users/nejeoui/Voxel/results/session-a-20260923
F() { grep -v "^Welcome\|^Have fun\|^AI agents"; }
for i in $(seq 1 90); do
  s=$($SSH 'test -f /root/voxel-session-a-collection.json && echo DONE || (tail -1 /root/voxel-session-a/gpu-run.log 2>/dev/null)' 2>/dev/null | F | tail -1)
  echo "$(date -u +%H:%M:%S) ${s:0:160}"
  [ "$s" = "DONE" ] && break
  sleep 60
done
[ "$s" = "DONE" ] || { echo "TIMEOUT waiting; instance NOT destroyed by finisher (watchdog 19:00Z remains)"; exit 1; }
$SCP root@71.17.164.141:/root/voxel-session-a-evidence.tar.gz root@71.17.164.141:/root/voxel-session-a-evidence.tar.receipt.json root@71.17.164.141:/root/voxel-session-a-collection.json root@71.17.164.141:/root/voxel-session-a-bootstrap.log $OUT/
want=$(python3 -c "import json;print(json.load(open('$OUT/voxel-session-a-evidence.tar.receipt.json'))['sha256'])")
got=$(shasum -a 256 $OUT/voxel-session-a-evidence.tar.gz | cut -d' ' -f1)
echo "archive want=$want got=$got"
if [ "$want" = "$got" ]; then
  echo "$(date -u +%FT%TZ) hash verified; destroying" | tee $OUT/destruction-request.log
  $SSH '/root/voxel-destroy.sh session-a-complete-archive-verified; cat /root/voxel-destroy.log' 2>&1 | F >> $OUT/destruction-request.log
  sleep 45
  if $SSH 'echo alive' 2>/dev/null | F | grep -q alive; then echo "WARNING: instance still reachable after destroy request" | tee -a $OUT/destruction-request.log
  else echo "$(date -u +%FT%TZ) instance unreachable after destroy request" | tee -a $OUT/destruction-request.log; fi
else
  echo "HASH MISMATCH: instance kept for re-download (watchdog 19:00Z remains)"; exit 2
fi
