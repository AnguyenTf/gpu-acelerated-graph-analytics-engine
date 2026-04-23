#include "triangle_count.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>
#include <iostream>

// Wraps any CUDA API call and checks for errors
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err));               \
        }                                                                    \
    } while (0)


/*  Example CSR:
    g = {
      num_vertices = 3,
      num_edges    = 3,
      row_ptr = {0, 2, 4, 6},
      col_idx = {1, 2, 0, 2, 0, 1}
  };*/

// ---------------- CPU VERSION ----------------

static int intersect_count_cpu_ordered(const CSRGraph& g, int u, int v);

long long triangle_count_cpu(const CSRGraph& g) {
    long long total = 0;

    // Loop over every vertex u
    for (int u = 0; u < g.num_vertices; ++u) {

        int start = g.row_ptr[u];
        int end   = g.row_ptr[u + 1];

        // Iterate over neighbors v of u
        for (int idx = start; idx < end; ++idx) {
            int v = g.col_idx[idx];

            // Only count triangles where u < v to avoid duplicates
            if (u < v) {
                total += intersect_count_cpu_ordered(g, u, v);
            }
        }
    }
    return total;
}

static int intersect_count_cpu_ordered(const CSRGraph& g, int u, int v) {
    int count = 0;

    int iu = g.row_ptr[u];
    int ju = g.row_ptr[u + 1];
    int iv = g.row_ptr[v];
    int jv = g.row_ptr[v + 1];

    // Two‑pointer intersection
    while (iu < ju && iv < jv) {
        int a = g.col_idx[iu];
        int b = g.col_idx[iv];

        // Skip neighbors <= v (canonical ordering)
        if (a <= v) { ++iu; continue; }
        if (b <= v) { ++iv; continue; }

        if (a == b) {
            ++count;
            ++iu;
            ++iv;
        } else if (a < b) {
            ++iu;
        } else {
            ++iv;
        }
    }
    return count;
}

// ---------------- GPU NAIVE VERSION ----------------

__device__
int intersect_count_gpu_ordered(const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                int u, int v);

long long triangle_count_gpu_naive(const CSRGraph& g, const EdgeListGraph& edges) {
    if (g.num_vertices == 0 || g.num_edges == 0) return 0;

    int* d_row_ptr      = nullptr;
    int* d_col_idx      = nullptr;
    int* d_edge_u       = nullptr;
    int* d_edge_v       = nullptr;
    long long* d_counts = nullptr;

    int num_vertices = g.num_vertices;
    int num_edges    = edges.num_edges;

    // Allocate CSR arrays on GPU
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, 2 * num_edges * sizeof(int)));

    // Copy CSR to GPU
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(),
                          (num_vertices + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(),
                          2 * num_edges * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Prepare edge lists for GPU
    std::vector<int> h_edge_u(num_edges), h_edge_v(num_edges);
    for (int i = 0; i < num_edges; ++i) {
        h_edge_u[i] = edges.edges[i].src;
        h_edge_v[i] = edges.edges[i].dst;
    }

    CUDA_CHECK(cudaMalloc(&d_edge_u, num_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_v, num_edges * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_edge_u, h_edge_u.data(),
                          num_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_v, h_edge_v.data(),
                          num_edges * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Allocate triangle counts
    CUDA_CHECK(cudaMalloc(&d_counts, num_edges * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_counts, 0, num_edges * sizeof(long long)));

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (num_edges + blockSize - 1) / blockSize;

    triangle_count_naive_kernel<<<gridSize, blockSize>>>(
        d_row_ptr, d_col_idx, d_edge_u, d_edge_v, num_edges, d_counts
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<long long> h_counts(num_edges);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                          num_edges * sizeof(long long),
                          cudaMemcpyDeviceToHost));

    long long total = 0;
    for (long long c : h_counts) total += c;

    // Free GPU memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_edge_u);
    cudaFree(d_edge_v);
    cudaFree(d_counts);

    return total;
}

// One thread per undirected edge (u < v)
__global__
void triangle_count_naive_kernel(const int* __restrict__ d_row_ptr,
                                 const int* __restrict__ d_col_idx,
                                 const int* __restrict__ d_edge_u,
                                 const int* __restrict__ d_edge_v,
                                 int num_edges,
                                 long long* __restrict__ d_counts) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;

    int u = d_edge_u[i];
    int v = d_edge_v[i];

    // Only count canonical edges
    if (u >= v) return;

    int local = intersect_count_gpu_ordered(d_row_ptr, d_col_idx, u, v);
    d_counts[i] = static_cast<long long>(local);
}

__device__
int intersect_count_gpu_ordered(const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                int u, int v) {

    int count = 0;

    int iu = row_ptr[u];
    int ju = row_ptr[u + 1];
    int iv = row_ptr[v];
    int jv = row_ptr[v + 1];

    while (iu < ju && iv < jv) {
        int a = col_idx[iu];
        int b = col_idx[iv];

        if (a <= v) { ++iu; continue; }
        if (b <= v) { ++iv; continue; }

        if (a == b) {
            ++count;
            ++iu;
            ++iv;
        } else if (a < b) {
            ++iu;
        } else {
            ++iv;
        }
    }
    return count;
}
