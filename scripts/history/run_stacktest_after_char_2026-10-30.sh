#!/usr/bin/env bash
# Wait for the aima character run to finish, then start the stacking test.
#
# Waiting on the log marker rather than a PID: train_char_2026-10-30.bat appends
# "==== ALL DONE" once train_sdxl.py exits, so this cannot hang on a dead process.
# The 60-minute cap stops a stale waiter from firing hours later.
set -u
cd /d/Gpart || exit 1

LOG=outputs/stacktest_2026-10-30/queue.log
CHARLOG=outputs/charrun_2026-10-30/console.log
mkdir -p outputs/stacktest_2026-10-30

echo "queue start $(date +%H:%M:%S)" >> "$LOG"

for i in $(seq 1 120); do
  grep -q "ALL DONE" "$CHARLOG" 2>/dev/null && break
  sleep 30
done

if ! grep -q "ALL DONE" "$CHARLOG" 2>/dev/null; then
  echo "TIMEOUT: char run never reported done $(date +%H:%M:%S)" >> "$LOG"
  exit 1
fi

echo "char run done, seen at $(date +%H:%M:%S)" >> "$LOG"
# let the training process actually release VRAM before the sampler allocates
sleep 30

bash scripts/stacktest_2026-10-30.sh >> outputs/stacktest_2026-10-30/console.log 2>&1
echo "stack test finished $(date +%H:%M:%S)" >> "$LOG"
