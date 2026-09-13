#!/usr/bin/env bash
# Paper shadow account daily runner (16:30 Mon-Fri, after the 15:30 data sync).
set -euo pipefail
cd /home/zxh/projects/3.qlib_ifind_beta

exec 9>/tmp/qlib_paper_shadow.lock
if ! flock -n 9; then
    echo "$(date '+%F %T') another paper_shadow instance holds the lock — skip"
    exit 0
fi

PY=/home/zxh/miniconda3/envs/qlib_ifind_beta/bin/python
echo "$(date '+%F %T') === paper_shadow day start ==="
"$PY" scripts/paper_shadow.py day --date "$(date +%F)"
echo "$(date '+%F %T') === paper_shadow day done ==="
