#!/bin/bash
# usage: eval/run_set.sh TAG SET [SET ...]   e.g.  eval/run_set.sh final-gpu16 heldout3
# Loads profiles/$PROFILE.env if PROFILE is set. Runs eval/run.py answers for each set.
set -e
cd "$(dirname "$0")/.."
if [ -n "$PROFILE" ]; then set -a; . "profiles/$PROFILE.env"; set +a; fi
tag=$1; shift
for s in "$@"; do uv run --frozen python eval/run.py answers --golden "eval/$s.yaml" --tag "$tag-$s"; done
echo DONE
