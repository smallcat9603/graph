#!/usr/bin/env bash
#
# fetch_stackoverflow.sh
#
# Downloads a Stack-Overflow Temporal dataset (SNAP) and converts it into
# the 3-column "src dst t" format expected by rw-temporal's data/ folder.
#
# Usage:
#   ./fetch_stackoverflow.sh [a2q|c2q|c2a|full]
#
#   a2q   answers-to-questions   (~17M edges, ~80MB gz / ~280MB raw)  -- default
#   c2q   comments-to-questions  (~20M edges, ~95MB gz / ~340MB raw)
#   c2a   comments-to-answers    (~25M edges, ~120MB gz / ~420MB raw)
#   full  union of all three     (~63M edges, ~250MB gz / ~1.8GB raw)
#
# Output is placed at data/<basename>.txt with timestamps normalised so
# the earliest interaction has t = 0. After this finishes you can run:
#
#   mpirun -np 4 ./rw <basename> <walkers> <steps> 1 50000
#
# e.g.
#   mpirun -np 4 ./rw stackoverflow_a2q 10000 20 1 50000
#
# Notes:
#   * In SNAP's sx-stackoverflow format the graph is directed (u -> v).
#     We load it via the same 3-column reader which treats every edge as
#     undirected; this is intentional for sampling experiments.
#   * Node ids in the raw file are sparse; partition_load_edgelist
#     densifies them on load, so no preprocessing of ids is needed.

set -euo pipefail

DATASET="${1:-a2q}"

case "$DATASET" in
    a2q)
        URL="https://snap.stanford.edu/data/sx-stackoverflow-a2q.txt.gz"
        NAME="stackoverflow_a2q"
        ;;
    c2q)
        URL="https://snap.stanford.edu/data/sx-stackoverflow-c2q.txt.gz"
        NAME="stackoverflow_c2q"
        ;;
    c2a)
        URL="https://snap.stanford.edu/data/sx-stackoverflow-c2a.txt.gz"
        NAME="stackoverflow_c2a"
        ;;
    full)
        URL="https://snap.stanford.edu/data/sx-stackoverflow.txt.gz"
        NAME="stackoverflow"
        ;;
    *)
        echo "unknown dataset: '$DATASET'" >&2
        echo "choose one of: a2q | c2q | c2a | full" >&2
        exit 1
        ;;
esac

# Resolve script directory and target data dir
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

GZ="${NAME}.txt.gz"
OUT="${NAME}.txt"

if [ -f "$OUT" ]; then
    echo "$OUT already exists. Delete it first if you want to re-fetch."
    exit 0
fi

# Step 1: download .gz
if [ ! -f "$GZ" ]; then
    echo "[1/3] downloading"
    echo "       $URL"
    if command -v wget >/dev/null 2>&1; then
        wget --no-check-certificate "$URL" -O "$GZ"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -k -o "$GZ" "$URL"
    else
        echo "neither wget nor curl is installed; cannot download" >&2
        exit 1
    fi
else
    echo "[1/3] $GZ already present, skipping download"
fi

# Step 2: scan for min/max timestamp (single decompress pass)
echo "[2/3] scanning for timestamp range"
read MIN_T MAX_T <<<"$(zcat "$GZ" | awk '
    NR == 1 { min = $3; max = $3 }
    $3 < min { min = $3 }
    $3 > max { max = $3 }
    END      { printf "%d %d\n", min, max }
')"
SPAN=$(( MAX_T - MIN_T ))
echo "       raw t in [$MIN_T, $MAX_T]  ->  output in [0, $SPAN]  (~$(( SPAN / 86400 )) days)"

# Step 3: normalise timestamps (subtract MIN_T) and emit final file
echo "[3/3] writing $OUT"
zcat "$GZ" | awk -v min="$MIN_T" '{printf "%d %d %d\n", $1, $2, $3 - min}' > "$OUT"

# Summary
LINES=$(wc -l < "$OUT")
MAX_NODE=$(awk '{
    if ($1 > m) m = $1
    if ($2 > m) m = $2
} END { print m }' "$OUT")

echo
echo "--- done ---"
echo "  file:        data/$OUT"
echo "  edges:       $LINES"
echo "  max node id: $MAX_NODE"
echo "  t range:     [0, $SPAN]  (~$(( SPAN / 86400 )) days)"
echo
echo "Next:"
echo "  cd .."
echo "  mpirun -np 4 ./rw $NAME 10000 20 1 50000"
