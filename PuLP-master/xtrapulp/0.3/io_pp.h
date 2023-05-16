/*
//@HEADER
// *****************************************************************************
//
//  XtraPuLP: Xtreme-Scale Graph Partitioning using Label Propagation
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/


#ifndef _IO_PP_H_
#define _IO_PP_H_

#include "xtrapulp.h"
#include "dist_graph.h"
#include "comms.h"

int load_graph_edges_32(char *input_filename, graph_gen_data_t *ggi, 
                        bool offset_vids);

int load_graph_edges_64(char *input_filename, graph_gen_data_t *ggi, 
                        bool offset_vids);

int scale_weights(uint64_t n, uint64_t num_weights, 
                  float* weights_in, int32_t* weights_out);

int read_adj(char* input_filename, 
  graph_gen_data_t *ggi, bool offset_vids);

int read_graph(char* input_filename, 
  graph_gen_data_t *ggi, bool offset_vids);

int exchange_edges(graph_gen_data_t *ggi, mpi_data_t* comm);

int exchange_edges_weighted(graph_gen_data_t *ggi, mpi_data_t* comm);

int output_parts(const char* filename, dist_graph_t* g, int32_t* parts);

int output_parts(const char* filename, dist_graph_t* g, 
                 int32_t* parts, bool offset_vids);

int read_parts(const char* filename, dist_graph_t* g, 
               pulp_data_t* pulp, bool offset_vids);

#endif
