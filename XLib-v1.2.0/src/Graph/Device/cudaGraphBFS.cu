#include "Graph/Device/cudaGraphBFS.cuh"

#include "Base/Host/fUtil.hpp"
#include "Base/Host/numeric.hpp"
#include "Base/Device/Util/cuda_util.cuh"

using namespace xlib;

namespace graph {

__constant__ _node_t* devF1 = 0;
__constant__ _node_t* devF2 = 0;
__constant__ _dist_t* devDistance = 0;

__device__ int devF2_size[4];

template<typename node_t, typename edge_t, typename cunode_t, typename dist_t>
const dist_t cudaGraphBFS<node_t, edge_t, cunode_t, dist_t>::INF_DIST =
                                             std::numeric_limits<dist_t>::max();

template<typename node_t, typename edge_t, typename cunode_t, typename dist_t>
cudaGraphBFS<node_t, edge_t, cunode_t, dist_t>
    ::cudaGraphBFS(GraphSTD<node_t, edge_t, dist_t>& _graph,
                   bool _inverse_graph,
                   unsigned _degree_options)
                   : cudaGraph<node_t, edge_t>
                     (_graph, _inverse_graph, _degree_options) {

    cudaMalloc(&cuDistance, static_cast<std::size_t>(V) * sizeof(dist_t));
    cudaMemcpyToSymbol(devDistance, &cuDistance, sizeof(dist_t*));
    __CUDA_ERROR("Graph Frontier Allocation");
}

template<typename node_t, typename edge_t, typename cunode_t, typename dist_t>
void cudaGraphBFS<node_t, edge_t, cunode_t, dist_t>::AllocateFrontiers() {
    std::size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::size_t frontier_size = (free / 2u) - 4 * (1024 * 1024);

    cudaMalloc(&cuF1, frontier_size);
    cudaMalloc(&cuF2, frontier_size);
    cudaMemcpyToSymbol(devF1, &cuF1, sizeof(node_t*));
    cudaMemcpyToSymbol(devF2, &cuF2, sizeof(node_t*));
    max_frontier_nodes = frontier_size / sizeof(node_t);

    __CUDA_ERROR("Graph Frontier Allocation");
    if (max_frontier_nodes < V)
        __ERROR("Device Memory not sufficient");
}

template<typename node_t, typename edge_t, typename cunode_t, typename dist_t>
cudaGraphBFS<node_t, edge_t, cunode_t, dist_t>::~cudaGraphBFS() {
    cudaFree(cuF1);
    cudaFree(cuF2);
    cudaFree(cuDistance);
    __CUDA_ERROR("Graph Free");
}

template<typename node_t, typename edge_t, typename cunode_t, typename dist_t>
void cudaGraphBFS<node_t, edge_t, cunode_t, dist_t>
    ::reset(int* Sources, int n_of_sources, bool update_distance) {

    cudaMemcpy(cuF1, Sources, n_of_sources * sizeof(node_t),
               cudaMemcpyHostToDevice);

    xlib::fill<<<Div(V, 128), 128>>>(cuDistance, V, INF_DIST);

    if (update_distance) {
        xlib::scatter <<<Div(n_of_sources, 128), 128>>>
            (cuF1, n_of_sources, cuDistance, dist_t(0));
    }
    int SizeArray[4] = {};
    cudaMemcpyToSymbol(devF2_size, SizeArray, sizeof(SizeArray));

    __CUDA_ERROR("Graph Reset");
}

template class cudaGraphBFS<>;

}   //@graph
