# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are informal labels for now since no public release has been cut.

Each "Fixed" entry below is written to be **individually revertible**, so it
can be reintroduced one-by-one in an ablation study to attribute the
measured speed-up to each change.

---

## [0.2.0] - 2026-05-25 -- "refactor + bug-fix baseline"

This release is a structural and correctness pass over the original
prototype. The random-walk algorithm, the MPI communication pattern, the
walker wire format, and the output log format are **unchanged**. Only
internals, module names, and the data-directory layout differ.

The static-graph baseline is now considered frozen; subsequent versions
will be the temporal-graph extension (see ARCHITECTURE.md "Planned
extensions").

### Added

- `config.h` -- single source of truth for the walker wire layout
  (`WALKER_HEADER_INTS`, accessor macros), default CLI arguments, and
  on-disk path roots (`DATA_DIR`, `LOG_DIR`).
- `intmap.{c,h}` -- a minimal open-addressing `int -> int` hash table
  shared by `routing` and `graph_io`.
- `path_buf_t` (in `walker.{c,h}`) -- exponential-growth buffer for
  completed walker paths; replaces the per-walker `realloc` of `paths`.
- `walker_t` -- a struct that carries the wire buffer alongside a cached
  `cur_local` field, eliminating per-step global-to-local reverse lookups.
- `README.md` and `docs/ARCHITECTURE.md`.

### Changed

- **Module renames (and split):**
  | Old           | New                          |
  | ------------- | ---------------------------- |
  | `algo.{c,h}`  | `walker.{c,h}`               |
  | `file.{c,h}`  | `graph_io.{c,h}`             |
  | `rt.{c,h}`    | `routing.{c,h}`              |
  | `rw.c`        | `main.c`                     |
  | (n/a)         | `config.h`, `intmap.{c,h}`   |
- **Function names** now describe responsibilities rather than file of
  origin, e.g. `get_rt -> routing_lookup`, `read_rt -> routing_load`,
  `map_nodes_in_edgelist -> partition_load_edgelist`, `gen_walker ->
  walker_spawn`, `check_graph -> partition_assert_connected`.
- **Walker buffer field access** is now via accessor macros
  (`WALKER_ID(buf)`, `WALKER_HOPS_OUT(buf)`, ...) instead of magic
  indices (`walker[0]`, `walker[3]`, ...). The wire format is byte-identical.
- **Data root** is now `data/` relative to the working directory at
  launch time. The legacy path `../../pyro/rw/data/...` is no longer
  used. Move (or symlink) inputs into `rw-temporal/data/` before running.
- **`igraph_neighbors` call site** updated for igraph 1.0's extended
  signature; passes `IGRAPH_LOOPS_TWICE` and `IGRAPH_MULTIPLE` to
  preserve pre-1.0 default semantics.

### Fixed

The performance-related entries describe bugs in the original code that
were behaviour-preserving (same output) but quietly expensive. Each is
labelled with an estimate of where it dominates so future ablation runs
can target meaningful workloads.

1. **F1 -- Per-step global-to-local reverse lookup.**
   `algo.c walk()` scanned `node_map[]` (`O(nnodes)`) on every hop after
   the first to recover `cur_local` from `cur_global`.
   *Now:* `walker_t` caches `cur_local`; the hash-backed `g2l` lookup
   runs only once, at walker arrival from `MPI_Recv`.
   *Dominates when:* `nnodes` per partition is large.

2. **F2 -- Per-step linear scan of the routing table.**
   `rt.c get_rt()` linear-scanned `dict[]` (`O(rt_size)`) on every step.
   *Now:* `routing_lookup` is `O(1)` via `intmap`.
   *Dominates when:* `rt_size` (number of boundary nodes) is large.

3. **F3 -- Quadratic node-id densification + double file pass.**
   `file.c map_nodes_in_edgelist()` ran two passes over the edgelist
   file and used `O(nnodes^2)` linear scans during the first pass.
   *Now:* hash-map-backed single pass into an `igraph_vector_int_t`,
   then `igraph_create` builds the graph in memory (no `.x.txt` temp
   file is written, and `remove()` after load is no longer needed).
   *Dominates when:* loading large partitions; also reduces disk I/O.

4. **F4 -- Per-step `realloc` of the walker buffer.**
   `algo.c walk()` grew the walker's int buffer by one element per step.
   *Now:* `walker_spawn` / `walker_adopt` allocates the full capacity
   (`WALKER_HEADER_INTS + nsteps`) once; receivers also grow inbound
   buffers to full size on arrival so subsequent steps do not realloc.
   *Dominates when:* `nsteps` is large.

5. **F5 -- Per-walker `realloc` of the completed-paths buffer.**
   `algo.c walk()` re-`realloc`ed `paths[]` for every completed walker.
   *Now:* `path_buf_t` doubles capacity on overflow, so total reallocs
   are `O(log n)`.
   *Dominates when:* `nwalkers_per_rank * size` is large.

6. **F6 -- Identical RNG seed across ranks initialised in the same second.**
   `rw.c` called `srand(time(NULL))`; all ranks starting in the same
   wall-clock second received the same seed and therefore the same
   random stream.
   *Now:* `srand(t ^ rank * 2654435761u)` mixes a Knuth multiplicative
   hash of the rank into the seed.
   *Affects:* statistical independence of walkers across ranks, not
   throughput. Strictly speaking a **correctness** fix rather than a
   performance one; included here because the legacy code path is now
   gone and downstream analyses should re-run.

7. **F7 -- Hard abort on malformed routing-table line.**
   `rt.c parseLine()` called `exit(0)` if the bracketed list was empty
   or unparseable, terminating the program rather than the parse.
   *Now:* the line is skipped with a warning and parsing continues.
   *Affects:* robustness on noisy or partially-empty routing files.

### Removed

- `algo.{c,h}`, `file.{c,h}`, `rt.{c,h}`, `rw.c` (replaced by the modules
  above).
- The intermediate `<base>.x.txt` densified-edgelist temp file written
  during graph load and deleted after `igraph_read_graph_edgelist`. The
  refactor builds the graph from an in-memory edge vector, so no temp
  file is created or removed.

### Build

- `Makefile` now compiles each module to a `.o` and links them; previously
  it linked the four `.c` files in one command. `make clean` removes both
  the binary and the object files.
- `-Wall -Wextra` are enabled in `CFLAGS`. The source compiles cleanly at
  this warning level under `mpicc` backed by `gcc`/`clang`.

### Compatibility notes

- The output log filename, content, and per-line field layout are
  unchanged. Existing analysis scripts targeting `log/*.txt` continue to
  work without modification.
- Input edgelist and routing-table file formats are unchanged. Only the
  data root directory moved.
- Re-running with the same `(dataset, nwalkers, nsteps, mode, np)` will
  produce **statistically equivalent but not bit-identical** logs: the
  RNG seed mixing in F6 changes the sampled paths.

---

## [0.1.0] - 2023-06-09 -- "original prototype" (retroactive)

Initial MPI random-walk implementation by the original author. Captured
here for reference; this version is no longer in the tree as of 0.2.0.

- Walker migration via blocking `MPI_Send` after each cross-partition hop.
- `MPI_Iprobe` busy loop with `MPI_Allreduce` termination detection.
- `MPI_Gatherv` of completed paths onto rank 0; one log file per run.
- Modules: `rw.c` (main), `algo.{c,h}` (walk loop), `file.{c,h}` (edgelist
  + node-id densification), `rt.{c,h}` (routing table).
- Known limitations addressed in 0.2.0 are listed under that release's
  "Fixed" section.
