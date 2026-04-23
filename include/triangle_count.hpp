#pragma once
#include "csr.hpp"
#include "graph.hpp"

/*  Triangle counting interfaces. 
    CPU: baseline implementation.
    GPU: implementation (one thread per edge).  */

// CPU baseline
long long triangle_count_cpu(const CSRGraph& g);

// GPU naive (one thread per edge)
long long triangle_count_gpu(const CSRGraph& g, const EdgeListGraph& edges);
