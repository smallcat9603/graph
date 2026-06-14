#!/bin/sh
#------ pjsub options (override node/proc on the command line per scaling point) ------#
#PJM -L rscgrp=regular-o
#PJM -L node=4
#PJM --mpi proc=192
#PJM -L elapse=0:20:00
#PJM -g GROUP            # <-- EDIT: your project/billing group code
#PJM -j
#PJM -N rwscale
#
# Strong/weak-scaling run for the distributed temporal-walk + co-shard embedding
# engine. Submit one job per scaling point, overriding node/proc:
#
#   pjsub -L node=2  --mpi proc=96   wisteria/job_scaling.sh
#   pjsub -L node=4  --mpi proc=192  wisteria/job_scaling.sh
#   pjsub -L node=8  --mpi proc=384  wisteria/job_scaling.sh
#   pjsub -L node=16 --mpi proc=768  wisteria/job_scaling.sh
#   pjsub -L node=32 --mpi proc=1536 wisteria/job_scaling.sh
#
# proc = node * 48 (flat MPI, one rank per A64FX core).  The dataset must be
# pre-partitioned into <proc> parts under data/<proc>/ (see README_wisteria.md).
# All paths are under /work (compute nodes cannot read /home).

DATASET=${DATASET:-stackoverflow_a2q}   # graph staged under data/<proc>/
NWALK=${NWALK:-50000}                    # walkers per rank  (weak scaling: fix this)
NSTEPS=${NSTEPS:-30}
DELTA_T=${DELTA_T:-0}                     # 0 = single-bucket batched (the real engine)
EMB=${EMB:-0}                             # 1 = also train co-shard embeddings
EMBMODE=${EMBMODE:-fused}                 # fused (local negs) | twostage (NOMAD-style)

P=${PJM_MPI_PROC}
echo "=== scaling: nodes=${PJM_NODE} proc=${P} dataset=${DATASET} nwalk/rank=${NWALK} emb=${EMB} mode=${EMBMODE} ==="

if [ "$EMB" = "1" ]; then
    # fused vs twostage: the comm comparison. PHASE reports emb_xchg (the remote
    # embedding comm two-stage pays and fused avoids) -- the gap grows on Tofu.
    EMBED_MODE=${EMBMODE} EMBED_DIM=64 EMBED_WIN=5 EMBED_NEG=5 \
        mpiexec -n ${P} ./rw ${DATASET} ${NWALK} ${NSTEPS} 0 ${DELTA_T}
else
    mpiexec -n ${P} ./rw ${DATASET} ${NWALK} ${NSTEPS} 0 ${DELTA_T}
fi
# Engine prints:  PHASE compute=.. exchange=.. allreduce=.. emb_xchg=.. comm_frac=..  + elapsed=..
