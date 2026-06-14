#!/bin/sh
# build_wisteria.sh -- compile the rw engine for Wisteria/BDEC-01 Odyssey
# (Fujitsu A64FX, aarch64). Run on a LOGIN node (cross-compiles to A64FX).
#
#   cd <repo>/rw-temporal && sh wisteria/build_wisteria.sh
#
# Produces ./rw (an aarch64 binary; will NOT run on the x86 login node, only
# inside an Odyssey batch/interactive job).
set -e

module load odyssey 2>/dev/null || { module load fj; module load fjmpi; }

# Fujitsu MPI C cross-compiler wrapper (-> A64FX). -Kfast is the Fujitsu opt set.
# If strdup() or other POSIX symbols are reported missing, add -D_POSIX_C_SOURCE=200809L.
# If trad-mode rejects flags, switch to clang mode: CFLAGS="-Nclang -Ofast".
make clean
make CC=mpifccpx CFLAGS="-Kfast -D_POSIX_C_SOURCE=200809L" LDFLAGS="-lm"

file ./rw   # should report ARM aarch64
echo "built ./rw for Odyssey (A64FX). Stage it + data under /work and submit via pjsub."
