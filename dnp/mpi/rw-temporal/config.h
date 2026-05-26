/*
 * config.h -- Compile-time constants, the on-wire walker layout, and a
 *             deterministic edge-timestamp synthesiser.
 *
 * A walker is exchanged between MPI ranks as a flat int[] buffer:
 *
 *   [0] id            -- unique walker id
 *   [1] start_ts      -- wall-clock when the walker was spawned
 *   [2] end_ts        -- wall-clock when the walker finished (0 until then)
 *   [3] hops_out      -- number of cross-partition migrations so far
 *   [4] t_cur         -- current temporal cursor; next hop must have t > t_cur
 *   [5 .. 5+k-1]      -- k visited global node ids (k grows with each step)
 *
 * Total capacity per walker = WALKER_HEADER_INTS + max_steps.
 *
 * Dead-end walkers (no valid future edge) are padded with WALKER_DEAD_END_PAD
 * in their remaining path slots so the on-disk log has fixed-width rows.
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

#define WALKER_HEADER_INTS 5

#define WALKER_ID(buf)        ((buf)[0])
#define WALKER_START_TS(buf)  ((buf)[1])
#define WALKER_END_TS(buf)    ((buf)[2])
#define WALKER_HOPS_OUT(buf)  ((buf)[3])
#define WALKER_TCUR(buf)      ((buf)[4])
#define WALKER_PATH(buf)      ((buf) + WALKER_HEADER_INTS)

/* Sentinel used to pad early-terminated walkers in the on-disk log. */
#define WALKER_DEAD_END_PAD (-1)

/* Initial value of t_cur on spawn. Chosen so that the first hop accepts
 * any edge (synthesised timestamps live in [0, TSYNTH_MAX), real-world
 * datasets normally have t >= 0 too). */
#define WALKER_INITIAL_TCUR (-1)

/* Defaults applied when the corresponding argv slot is missing. */
#define DEFAULT_DATASET   "facebook"
#define DEFAULT_NWALKERS  1
#define DEFAULT_NSTEPS    80
#define DEFAULT_MODE      0   /* 0 = partitioned, 1 = full graph per rank */

/* Paths are interpreted relative to the directory rw is launched from. */
#define DATA_DIR "data"
#define LOG_DIR  "log"

/* MPI message tag for in-flight walkers. */
#define TAG_WALKER 0

/* Synthesised edge timestamps live in [0, TSYNTH_MAX). The space is small
 * enough to be human-readable in logs but large enough that collisions are
 * statistically rare for graphs up to ~10^5 edges. */
#define TSYNTH_MAX 1000000

/* Deterministic per-edge timestamp synthesised from the (src,dst) pair.
 * Symmetric in (a,b) so an undirected edge gets the same t regardless of
 * which endpoint is listed first; consistent across the local TAL and the
 * routing table since both sides hash on (global_id, global_id). */
static inline int synth_timestamp(int a, int b) {
    if (a > b) { int tmp = a; a = b; b = tmp; }
    uint32_t h = (uint32_t) a * 2654435761u;
    h ^= (uint32_t) b * 40503u;
    h += 0x9e3779b9u;
    return (int) (h % TSYNTH_MAX);
}

#endif /* CONFIG_H */
