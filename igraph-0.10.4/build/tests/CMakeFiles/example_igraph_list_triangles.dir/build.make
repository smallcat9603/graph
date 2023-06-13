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
include tests/CMakeFiles/example_igraph_list_triangles.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/example_igraph_list_triangles.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/example_igraph_list_triangles.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/example_igraph_list_triangles.dir/flags.make

tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o: tests/CMakeFiles/example_igraph_list_triangles.dir/flags.make
tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o: /github/graph/igraph-0.10.4/examples/simple/igraph_list_triangles.c
tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o: tests/CMakeFiles/example_igraph_list_triangles.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/github/graph/igraph-0.10.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o"
	cd /github/graph/igraph-0.10.4/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o -MF CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o.d -o CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o -c /github/graph/igraph-0.10.4/examples/simple/igraph_list_triangles.c

tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.i"
	cd /github/graph/igraph-0.10.4/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /github/graph/igraph-0.10.4/examples/simple/igraph_list_triangles.c > CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.i

tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.s"
	cd /github/graph/igraph-0.10.4/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /github/graph/igraph-0.10.4/examples/simple/igraph_list_triangles.c -o CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.s

# Object files for target example_igraph_list_triangles
example_igraph_list_triangles_OBJECTS = \
"CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o"

# External object files for target example_igraph_list_triangles
example_igraph_list_triangles_EXTERNAL_OBJECTS =

tests/example_igraph_list_triangles: tests/CMakeFiles/example_igraph_list_triangles.dir/__/examples/simple/igraph_list_triangles.c.o
tests/example_igraph_list_triangles: tests/CMakeFiles/example_igraph_list_triangles.dir/build.make
tests/example_igraph_list_triangles: src/libigraph.a
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libm.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libblas.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libf77blas.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libatlas.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/liblapack.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libblas.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libf77blas.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libatlas.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/liblapack.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libxml2.so
tests/example_igraph_list_triangles: /usr/lib/gcc/aarch64-linux-gnu/9/libgomp.so
tests/example_igraph_list_triangles: /usr/lib/aarch64-linux-gnu/libpthread.so
tests/example_igraph_list_triangles: tests/CMakeFiles/example_igraph_list_triangles.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/github/graph/igraph-0.10.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_igraph_list_triangles"
	cd /github/graph/igraph-0.10.4/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_igraph_list_triangles.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/example_igraph_list_triangles.dir/build: tests/example_igraph_list_triangles
.PHONY : tests/CMakeFiles/example_igraph_list_triangles.dir/build

tests/CMakeFiles/example_igraph_list_triangles.dir/clean:
	cd /github/graph/igraph-0.10.4/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/example_igraph_list_triangles.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/example_igraph_list_triangles.dir/clean

tests/CMakeFiles/example_igraph_list_triangles.dir/depend:
	cd /github/graph/igraph-0.10.4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /github/graph/igraph-0.10.4 /github/graph/igraph-0.10.4/tests /github/graph/igraph-0.10.4/build /github/graph/igraph-0.10.4/build/tests /github/graph/igraph-0.10.4/build/tests/CMakeFiles/example_igraph_list_triangles.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/example_igraph_list_triangles.dir/depend

