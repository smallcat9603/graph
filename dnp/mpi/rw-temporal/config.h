/*
 * config.h -- Compile-time constants and the on-wire walker layout.
 *
 * A walker is exchanged between MPI ranks as a flat int[] buffer:
 *
 *   [0] id            -- unique walker id
 *   [1] start_ts      -- wall-clock when the walker was spawned
 *   [2] end_ts        -- wall-clock when the walker finished (0 until then)
 *   [3] hops_out      -- number of cross-partition migrations so far
 *   [4 .. 4+k-1]      -- k visited global node ids (k grows with each step)
 *
 * Total capacity per walker = WALKER_HEADER_INTS + max_steps.
 */
#ifndef CONFIG_H
#define CONFIG_H

#define WALKER_HEADER_INTS 4

#define WALKER_ID(buf)        ((buf)[0])
#define WALKER_START_TS(buf)  ((buf)[1])
#define WALKER_END_TS(buf)    ((buf)[2])
#define WALKER_HOPS_OUT(buf)  ((buf)[3])
#define WALKER_PATH(buf)      ((buf) + WALKER_HEADER_INTS)

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

#endif /* CONFIG_H */
