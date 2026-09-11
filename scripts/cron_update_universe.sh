#!/usr/bin/env bash
# 883926 universe snapshot pre-market cron (08:30 Mon-Fri, universe-only).
# Materialization/retraining stay manual by project decision (2026-09-11).
set -euo pipefail
cd /home/zxh/projects/3.qlib_ifind_beta

# one instance at a time (manual runs / overlapping crons)
exec 9>/tmp/qlib_universe_cron.lock
if ! flock -n 9; then
    echo "$(date '+%F %T') another update_universe instance holds the lock — skip"
    exit 0
fi

PY=/home/zxh/miniconda3/envs/qlib_ifind_beta/bin/python
echo "$(date '+%F %T') === update_universe start ==="
rc=0
"$PY" -m scripts.update_universe --end "$(date +%F)" || rc=$?
echo "$(date '+%F %T') === update_universe done (rc=$rc) ==="
exit "$rc"
