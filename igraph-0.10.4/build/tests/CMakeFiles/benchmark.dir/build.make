# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /github/graph/igraph-0.10.4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /github/graph/igraph-0.10.4/build

# Utility rule file for benchmark.

# Include any custom commands dependencies for this target.
include tests/CMakeFiles/benchmark.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/benchmark.dir/progress.make

tests/CMakeFiles/benchmark:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/github/graph/igraph-0.10.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running benchmarks..."
	cd /github/graph/igraph-0.10.4/build/tests && true

benchmark: tests/CMakeFiles/benchmark
benchmark: tests/CMakeFiles/benchmark.dir/build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_average_path_length_unweighted"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_average_path_length_unweighted
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_betweenness"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_betweenness
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_betweenness_weighted"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_betweenness_weighted
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_cliques"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_cliques
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_closeness_weighted"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_closeness_weighted
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_coloring"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_coloring
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_decompose"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_decompose
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_degree"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_degree
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_distances"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_distances
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_ecc"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_ecc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_layout_umap"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_layout_umap
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_matrix_transpose"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_matrix_transpose
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_maximal_cliques"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_maximal_cliques
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_neighborhood"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_neighborhood
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_pagerank"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_pagerank
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_pagerank_weighted"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_pagerank_weighted
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_power_law_fit"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_power_law_fit
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_qsort"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_qsort
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_random_walk"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_random_walk
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_transitivity"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_transitivity
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: igraph_voronoi"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_igraph_voronoi
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running benchmark: inc_vs_adj"
	cd /github/graph/igraph-0.10.4/build/tests && ./benchmark_inc_vs_adj
.PHONY : benchmark

# Rule to build all files generated by this target.
tests/CMakeFiles/benchmark.dir/build: benchmark
.PHONY : tests/CMakeFiles/benchmark.dir/build

tests/CMakeFiles/benchmark.dir/clean:
	cd /github/graph/igraph-0.10.4/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/benchmark.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/benchmark.dir/clean

tests/CMakeFiles/benchmark.dir/depend:
	cd /github/graph/igraph-0.10.4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /github/graph/igraph-0.10.4 /github/graph/igraph-0.10.4/tests /github/graph/igraph-0.10.4/build /github/graph/igraph-0.10.4/build/tests /github/graph/igraph-0.10.4/build/tests/CMakeFiles/benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/benchmark.dir/depend
