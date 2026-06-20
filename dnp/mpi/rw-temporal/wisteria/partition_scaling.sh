#!/bin/sh
# partition_scaling.sh -- run on the LOCAL Mac (.venv_part has a working pymetis;
# the cluster's Python 3.6 cannot build it). Partitions a dataset into every
# proc count of the scaling sweep, ready to rsync to /work on Wisteria.
#
#   sh wisteria/partition_scaling.sh                       # stackoverflow_a2q, default procs
#   sh wisteria/partition_scaling.sh tgbl_review 96 192    # custom dataset / procs
#
# proc = nodes*48 (flat MPI). Default sweep = 2,4,8,16,32 nodes.
# Heavy: each large-graph partition is ~minutes + ~2 GB RAM; 1536 parts => 3072
# files. Skips proc counts already partitioned so the run is resumable.
set -e
PY=${PY:-.venv_part/bin/python}
DS=${1:-stackoverflow_a2q}
shift 2>/dev/null || true
PROCS="${*:-96 192 384 768 1536}"

for P in $PROCS; do
  if [ -f "data/$P/$DS.sub0.txt" ] || [ -f "data/$P/$DS.sub0.txt.part000" ]; then
    echo "=== $DS / $P parts: already present, skip ==="
    continue
  fi
  echo "=== partitioning $DS into $P parts ==="
  $PY partition_metis.py "data/$DS.txt" "$P"
done

echo
echo "done. Upload the partition dirs to /work on Wisteria, e.g.:"
echo "  rsync -av data/96 data/192 data/384 data/768 data/1536 \\"
echo "      z30130@wisteria.cc.u-tokyo.ac.jp:/work/gz00/z30130/<rw-temporal>/data/"
echo "(also upload data/<DS>.txt itself if not already there)"
