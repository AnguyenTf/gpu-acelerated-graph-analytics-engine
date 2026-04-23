#pragma once
#include <vector>
#include "graph.hpp"

/* CSR (Compressed Sparse Row) representation of an undirected graph.
   Each undirected edge appears twice in col_idx (u→v and v→u). */

struct CSRGraph {
    int num_vertices{};          // number of vertices
    int num_edges{};             // number of undirected edges
    std::vector<int> row_ptr;    // size: num_vertices + 1
    std::vector<int> col_idx;    // size: 2 * num_edges
};

// Build a CSR graph from an edge list (undirected, src < dst).
CSRGraph build_csr_from_edge_list(const EdgeListGraph& g);
