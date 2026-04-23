# Triangle Counting in CUDA & C++

* Insert Triangle Count Architecture *

## Introducing Triangle Counting for Graph Analytics in CUDA & C++

In this repository, I designed and implemented a fully high performance GPU implementation of triangle counting using CUDA and C++, built from the ground up with a custom graph loader, CSR converter, and GPU kernel. This project demonstrates how to design a complete graph anlytics pipeline optimized for parallel execution on NVIDIA GPUs.

---

### Why this project matters 

I built this project because I wanted to understand what really goes into GPU-accelerated graph analytics, and not just writing a CUDA kernel, but everything around it. I thought this project turned out to be the perfect challenge: it's simple to describe, but it forces me to think carefully about data layout, memory access patterns, and how GPUs behave when the work isn't perfectly regular. Working through this from scratch taught me a lot about how high performnace systems are built. I had to design a graph loader that doesn't fall apart on a messy input, build a CSR converter that produces clean and sorted adjacency lists, and then write a kernel that can handle thousands of edges in parallel. 

---

### Highlights

- Full custom graph ingestion pipeline
- Clean CSR construction with sorted adjacency lists
- Memory coalesced structure of arrays layout for edges
- Safe CUDA error handing with CUDA_CHECK
- Per-edge parallelism for high GPU throughput

---
### Helpful References

- [https://www.nvidia.com/en-us/glossary/graph-analytics/

---

### Project Structure

```
accelerated_graph_analytics/
├── data/
│   ├── sample_graph.txt
├── include/
│   ├── csr.hpp
│   ├── graph.hpp
│   ├── triangle_count.hpp
├── src/
│   ├── csr_converter.cpp
│   ├── graph_loader.cpp
│   ├── main.cpp
│   └── triangle_count.cu
├── triangle_count.exe
```
 
### Architecture

1) Read an edge list file (ordered or unordered)
2) Normalize each edge so that u < v
3) Sort edges by (u, v) to guarantee deterministic adjacency lists
4) Build CSR (row_ptr + col_idx)
5) For each edge (u, v), run a two-pointer intersection on neighbors(u) and neighbors(v)
6) Count all w such that u < v < w
7) Parallelize the entire process on the GPU
