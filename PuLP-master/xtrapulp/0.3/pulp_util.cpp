
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
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <assert.h>

#include "xtrapulp.h"
#include "pulp_util.h"
#include "pulp_data.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;


int part_eval(dist_graph_t* g, pulp_data_t* pulp)
{
  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    pulp->part_vert_sizes[i] = 0;
    pulp->part_edge_sizes[i] = 0;
    pulp->part_cut_sizes[i] = 0;
  }
  pulp->cut_size = 0;
  pulp->max_cut = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];
    ++pulp->part_vert_sizes[part];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    pulp->part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      if (part_out != part)
      {
        ++pulp->part_cut_sizes[part];
        ++pulp->cut_size;
      }
    } 
  }

  MPI_Allreduce(MPI_IN_PLACE, pulp->part_vert_sizes, pulp->num_parts, 
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;

  uint64_t global_ghost = g->n_ghost;
  uint64_t max_ghost = 0;
  MPI_Allreduce(&global_ghost, &max_ghost, 1,
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_ghost, 1,
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  double ghost_balance = 
    (double)max_ghost / ((double)global_ghost / (double)pulp->num_parts);

  int64_t max_v_size = 0;
  int32_t max_v_part = -1;
  int64_t max_e_size = 0;
  int32_t max_e_part = -1;
  int64_t max_c_size = 0;
  int32_t max_c_part = -1;
  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    if (max_v_size < pulp->part_vert_sizes[i])
    {
      max_v_size = pulp->part_vert_sizes[i];
      max_v_part = i;
    }
    if (max_e_size < pulp->part_edge_sizes[i])
    {
      max_e_size = pulp->part_edge_sizes[i];
      max_e_part = i;
    }
    if (max_c_size < pulp->part_cut_sizes[i])
    {
      max_c_size = pulp->part_cut_sizes[i];
      max_c_part = i;
    }
  }

  pulp->max_v = (double)max_v_size / pulp->avg_vert_size;
  pulp->max_e = (double)max_e_size / pulp->avg_edge_size;
  pulp->max_c = (double)max_c_size / ((double)pulp->cut_size / (double)pulp->num_parts) / 2.0;
  pulp->max_cut = max_c_size;

  if (procid == 0)
  {
    printf("---------------------------------------------------------\n");
    printf("EdgeCut: %li, MaxPartCut: %li\nVertexBalance: %2.3lf (%d, %li), EdgeBalance: %2.3lf (%d, %li)\nCutBalance: %2.3lf (%d, %li), GhostBalance: %2.3lf (%li)\n", 
      pulp->cut_size, pulp->max_cut,
      pulp->max_v, max_v_part, max_v_size,
      pulp->max_e, max_e_part, max_e_size,
      pulp->max_c, max_c_part, max_c_size,
      ghost_balance, max_ghost);
    printf("---------------------------------------------------------\n");
    for (int32_t i = 0; i < pulp->num_parts; ++i)
      printf("Part: %d, VertSize: %li, EdgeSize: %li, Cut: %li\n",
      i, pulp->part_vert_sizes[i], pulp->part_edge_sizes[i], 
      pulp->part_cut_sizes[i]);
    printf("---------------------------------------------------------\n");
  }

  return 0;
} 

int part_eval(dist_graph_t* g, int32_t* parts, int32_t num_parts)
{
  pulp_data_t pulp;
  init_pulp_data(g, &pulp, num_parts);
  memcpy(parts, pulp.local_parts, g->n_local*sizeof(int32_t));
  
  for (int32_t i = 0; i < pulp.num_parts; ++i)
  {
    pulp.part_sizes[0][i] = 0;
    pulp.part_edge_sizes[i] = 0;
    pulp.part_cut_sizes[i] = 0;
  }
  pulp.cut_size = 0;
  pulp.max_cut = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp.local_parts[vert_index];
    ++pulp.part_sizes[0][part];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    pulp.part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp.local_parts[out_index];
      if (part_out != part)
      {
        ++pulp.part_cut_sizes[part];
        ++pulp.cut_size;
      }
    } 
  }

  MPI_Allreduce(MPI_IN_PLACE, pulp.part_vert_sizes, pulp.num_parts, 
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp.part_edge_sizes, pulp.num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp.part_cut_sizes, pulp.num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp.cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp.cut_size /= 2;

  uint64_t global_ghost = g->n_ghost;
  uint64_t max_ghost = 0;
  MPI_Allreduce(&global_ghost, &max_ghost, 1,
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_ghost, 1,
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  double ghost_balance = 
    (double)max_ghost / ((double)global_ghost / (double)pulp.num_parts);

  int64_t max_v_size = 0;
  int32_t max_v_part = -1;
  int64_t max_e_size = 0;
  int32_t max_e_part = -1;
  int64_t max_c_size = 0;
  int32_t max_c_part = -1;
  for (int32_t i = 0; i < pulp.num_parts; ++i)
  {
    if (max_v_size < pulp.part_vert_sizes[i])
    {
      max_v_size = pulp.part_vert_sizes[i];
      max_v_part = i;
    }
    if (max_e_size < pulp.part_edge_sizes[i])
    {
      max_e_size = pulp.part_edge_sizes[i];
      max_e_part = i;
    }
    if (max_c_size < pulp.part_cut_sizes[i])
    {
      max_c_size = pulp.part_cut_sizes[i];
      max_c_part = i;
    }
  }

  pulp.max_v = (double)max_v_size / ((double)g->n / (double)pulp.num_parts);
  pulp.max_e = (double)max_e_size / ((double)g->m / (double)pulp.num_parts);
  pulp.max_c = (double)max_c_size / ((double)pulp.cut_size / (double)pulp.num_parts) / 2.0;
  pulp.max_cut = max_c_size;

  if (procid == 0)
  {
    printf("EVAL ec: %li, vb: %2.3lf (%d, %li), eb: %2.3lf (%d, %li), cb: %2.3lf (%d, %li), gb: %2.3lf (%li)\n", 
      pulp.cut_size, 
      pulp.max_v, max_v_part, max_v_size,
      pulp.max_e, max_e_part, max_e_size,
      pulp.max_c, max_c_part, max_c_size,
      ghost_balance, max_ghost);

    for (int32_t i = 0; i < pulp.num_parts; ++i)
      printf("p: %d, v: %li, e: %li, cut: %li\n",
      i, pulp.part_vert_sizes[i], pulp.part_edge_sizes[i], 
      pulp.part_cut_sizes[i]);
  }

  clear_pulp_data(&pulp);

  return 0;
} 


int part_eval_weighted(dist_graph_t* g, pulp_data_t* pulp)
{
  for (int32_t p = 0; p < pulp->num_parts; ++p)
  {
    for (uint64_t w = 0; w < g->num_vert_weights; ++w)
      pulp->part_sizes[w][p] = 0;
    pulp->part_cut_sizes[p] = 0;
  }
  pulp->cut_size = 0;
  pulp->max_cut = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];
    for (uint64_t w = 0; w < g->num_vert_weights; ++w)
      pulp->part_sizes[w][part] += 
          g->vert_weights[vert_index*g->num_vert_weights + w];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    int32_t* weights = out_weights(g, vert_index);
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      if (part_out != part)
      {
        pulp->part_cut_sizes[part] += weights[j];
        pulp->cut_size += weights[j];
      }
    }
  }

  for (uint64_t w = 0; w < g->num_vert_weights; ++w) {
    MPI_Allreduce(MPI_IN_PLACE, pulp->part_sizes[w], pulp->num_parts, 
      MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  }
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;

  int64_t* part_size_sums = (int64_t*)malloc(g->num_vert_weights*sizeof(int64_t));
  for (uint64_t w = 0; w < g->num_vert_weights; ++w)
    part_size_sums[w] = 0;
  int64_t part_cut_size_sum = 0;

  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    for (uint64_t w = 0; w < g->num_vert_weights; ++w)
      part_size_sums[w] += pulp->part_sizes[w][i];
    part_cut_size_sum += pulp->part_cut_sizes[i];

    if (pulp->part_cut_sizes[i] > pulp->max_cut)
      pulp->max_cut = pulp->part_cut_sizes[i];
  }

  uint64_t global_ghost = g->n_ghost;
  uint64_t max_ghost = 0;
  MPI_Allreduce(&global_ghost, &max_ghost, 1,
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_ghost, 1,
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  double ghost_balance = 
    (double)max_ghost / ((double)global_ghost / (double)pulp->num_parts);

  double* max_overweights = (double*)malloc(g->num_vert_weights*sizeof(double));
  int64_t* max_sizes = (int64_t*)malloc(g->num_vert_weights*sizeof(int64_t));
  int32_t* max_sizes_part = (int32_t*)malloc(g->num_vert_weights*sizeof(int32_t));

  for (uint64_t w = 0; w < g->num_vert_weights; ++w) {
    max_sizes[w] = 0;
    max_sizes_part[w] = -1;
    for (int32_t i = 0; i < pulp->num_parts; ++i) {
      if (max_sizes[w] < pulp->part_sizes[w][i]) {
        max_sizes[w] = pulp->part_sizes[w][i];
        max_sizes_part[w] = i;
      }
    }
    max_overweights[w] = (double)max_sizes[w] / 
        ((double)part_size_sums[w] / (double)pulp->num_parts);
  }

  int64_t max_c_size = 0;
  int32_t max_c_part = -1;
  for (int32_t i = 0; i < pulp->num_parts; ++i) {
    if (max_c_size < pulp->part_cut_sizes[i]) {
      max_c_size = pulp->part_cut_sizes[i];
      max_c_part = i;
    }
  }
  pulp->max_c = 
      (double)max_c_size / ((double)pulp->cut_size / (double)pulp->num_parts);

  if (procid == 0) {
    printf("EVAL: %li, ", pulp->cut_size);
    for (uint64_t w = 0; w < g->num_vert_weights; ++w) {
      printf("%lu: %2.3lf (%d, %li), ", 
        w, max_overweights[w], max_sizes_part[w], max_sizes[w]);
    }
    printf("cb: %2.3lf (%d, %li), gb: %2.3lf (%li)\n",
      pulp->max_c, max_c_part, max_c_size,
      ghost_balance, max_ghost);

    if (debug) {
      for (int32_t p = 0; p < pulp->num_parts; ++p) {
        printf("p: %d, ", p);
        for (uint64_t w = 0; w < g->num_vert_weights; ++w) {
          printf("%lu: %lu, ", w, pulp->part_sizes[w][p]);
        }
        printf("\n");
      }
    } 
  }
  
  free(part_size_sums);
  free(max_overweights);
  free(max_sizes_part);
  free(max_sizes);

  return 0;
} 
