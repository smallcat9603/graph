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

# Include any dependencies generated for this target.
include tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/flags.make

tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o: tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/flags.make
tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o: /github/graph/igraph-0.10.4/tests/benchmarks/igraph_betweenness_weighted.c
tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o: tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/github/graph/igraph-0.10.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o"
	cd /github/graph/igraph-0.10.4/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o -MF CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o.d -o CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o -c /github/graph/igraph-0.10.4/tests/benchmarks/igraph_betweenness_weighted.c

tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.i"
	cd /github/graph/igraph-0.10.4/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /github/graph/igraph-0.10.4/tests/benchmarks/igraph_betweenness_weighted.c > CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.i

tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.s"
	cd /github/graph/igraph-0.10.4/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /github/graph/igraph-0.10.4/tests/benchmarks/igraph_betweenness_weighted.c -o CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.s

# Object files for target benchmark_igraph_betweenness_weighted
benchmark_igraph_betweenness_weighted_OBJECTS = \
"CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o"

# External object files for target benchmark_igraph_betweenness_weighted
benchmark_igraph_betweenness_weighted_EXTERNAL_OBJECTS =

tests/benchmark_igraph_betweenness_weighted: tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/benchmarks/igraph_betweenness_weighted.c.o
tests/benchmark_igraph_betweenness_weighted: tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/build.make
tests/benchmark_igraph_betweenness_weighted: src/libigraph.a
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libm.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libblas.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libf77blas.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libatlas.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/liblapack.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libblas.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libf77blas.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libatlas.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/liblapack.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libxml2.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/gcc/aarch64-linux-gnu/9/libgomp.so
tests/benchmark_igraph_betweenness_weighted: /usr/lib/aarch64-linux-gnu/libpthread.so
tests/benchmark_igraph_betweenness_weighted: tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/github/graph/igraph-0.10.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable benchmark_igraph_betweenness_weighted"
	cd /github/graph/igraph-0.10.4/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_igraph_betweenness_weighted.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/build: tests/benchmark_igraph_betweenness_weighted
.PHONY : tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/build

tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/clean:
	cd /github/graph/igraph-0.10.4/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_igraph_betweenness_weighted.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/clean

tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/depend:
	cd /github/graph/igraph-0.10.4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /github/graph/igraph-0.10.4 /github/graph/igraph-0.10.4/tests /github/graph/igraph-0.10.4/build /github/graph/igraph-0.10.4/build/tests /github/graph/igraph-0.10.4/build/tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/benchmark_igraph_betweenness_weighted.dir/depend

