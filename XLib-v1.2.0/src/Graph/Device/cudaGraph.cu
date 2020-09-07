/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "Graph/Device/cudaGraph.cuh"
#include "Base/Device/Util/cuda_util.cuh"

namespace graph {

__constant__ _node_t    devV;
__constant__ _cuedge_t  devE;
__constant__ _cunode_t*  devOutOffset;
__constant__ _cuedge_t*  devOutEdges;
__constant__ _cunode_t*  devInOffset;
__constant__ _cuedge_t*  devInEdges;
__constant__ _degree2_t* devInOutDegrees;
__constant__ _degree_t*  devOutDegrees;
__constant__ _degree_t*  devInDegrees;

template<typename node_t, typename edge_t, typename cunode_t, typename dist>
cudaGraph<node_t, edge_t, cunode_t, dist>
    ::cudaGraph(GraphSTD<node_t, edge_t, dist>& _graph, bool _inverse_graph,
                unsigned _degree_options) : graph(_graph) {

    V = graph.V;
    E = graph.E;
    inverse_graph = _inverse_graph;
    degree_options = _degree_options;

    cudaMemcpyToSymbol(devV, &V, sizeof(node_t));
    cudaMemcpyToSymbol(devE, &E, sizeof(edge_t));

    cudaMalloc(&cuOutOffset, V * sizeof(cunode_t));
    cudaMalloc(&cuOutEdges, E * sizeof(edge_t));
    cudaMemcpyToSymbol(devOutOffset, &cuOutOffset, sizeof(cunode_t*));
    cudaMemcpyToSymbol(devOutEdges, &cuOutEdges, sizeof(edge_t*));
    __CUDA_ERROR("CUDA -------- Allocation");
    if (inverse_graph) {
        cudaMalloc(&cuInOffset, V * sizeof(cunode_t));
        cudaMalloc(&cuInEdges, E * sizeof(edge_t));
        cudaMemcpyToSymbol(devInOffset, &cuInOffset, sizeof(cunode_t*));
        cudaMemcpyToSymbol(devInEdges, &cuInEdges, sizeof(edge_t*));
    }

    if ((degree_options & IN_DEGREE) && (degree_options & OUT_DEGREE)) {
        cudaMalloc(&cuInOutDegrees, V * sizeof(degree2_t));
        cudaMemcpyToSymbol(devInOutDegrees, &cuInOutDegrees,sizeof(degree2_t*));
    }
    else if (degree_options & IN_DEGREE) {
        cudaMalloc(&cuInDegrees, V * sizeof(degree_t));
        cudaMemcpyToSymbol(devInDegrees, &cuInDegrees, sizeof(degree_t*));
    }
    else if (degree_options & OUT_DEGREE) {
        cudaMalloc(&cuOutDegrees, V * sizeof(degree_t));
        cudaMemcpyToSymbol(devOutDegrees, &cuOutDegrees, sizeof(degree_t*));
    }
    __CUDA_ERROR("CUDA Graph Allocation");

    //==========================================================================

    cunode_t* tmp_OutOffset = new cunode_t[V];
    for (node_t i = 0; i < V; i++)
        tmp_OutOffset[i] = make_int2(graph.OutOffset[i],
                                     graph.OutOffset[i + 1]);

    cudaMemcpy(cuOutOffset, tmp_OutOffset, V * sizeof(cunode_t),
               cudaMemcpyHostToDevice);

    delete[] tmp_OutOffset;

    //--------------------------------------------------------------------------
    /*edge_t* tmp_OutEdges = new edge_t[E];
    for (edge_t i = 0; i < E; i++)
        tmp_OutEdges[i] = edge_t(graph.OutEdges[i], graph.OutDegrees[i]);

    cudaMemcpy(devGraph.devOutEdges, tmp_OutEdges, E * sizeof(edge_t),
               cudaMemcpyHostToDevice);

    delete[] tmp_OutEdges;*/

    cudaMemcpy(cuOutEdges, graph.OutEdges, E * sizeof(edge_t),
               cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------

    if (inverse_graph) {
        cunode_t* tmp_InOffset = new cunode_t[V];
        for (node_t i = 0; i < V; i++)
            tmp_InOffset[i] = make_int2(graph.InOffset[i],
                                        graph.InOffset[i + 1]);

        cudaMemcpy(cuInOffset, tmp_InOffset,
                   V * sizeof(cunode_t), cudaMemcpyHostToDevice);

        delete[] tmp_InOffset;

        cudaMemcpy(cuInEdges, graph.InEdges,
                   E * sizeof(edge_t), cudaMemcpyHostToDevice);
    }

    if ((degree_options & IN_DEGREE) && (degree_options & OUT_DEGREE)) {
        degree2_t* tmpInOutDegrees = new degree2_t[V];
        for (node_t i = 0; i < V; i++)
            tmpInOutDegrees[i] = make_int2(graph.InDegrees[i],
                                           graph.OutDegrees[i]);

        cudaMemcpy(cuInOutDegrees, tmpInOutDegrees, V * sizeof(degree2_t),
                   cudaMemcpyHostToDevice);
        delete[] tmpInOutDegrees;
    }
    else if (degree_options & IN_DEGREE) {
        cudaMemcpy(cuInDegrees, graph.InDegrees, V * sizeof(degree_t),
                   cudaMemcpyHostToDevice);
    } else if (degree_options & OUT_DEGREE) {
        cudaMemcpy(cuOutDegrees, graph.OutDegrees, V * sizeof(degree_t),
                   cudaMemcpyHostToDevice);
    }
    __CUDA_ERROR("Graph Copy To Device");
}

template<typename node_t, typename edge_t, typename cunode_t, typename dist_t>
cudaGraph<node_t, edge_t, cunode_t, dist_t>::~cudaGraph() {
    cudaFree(cuOutOffset);
    cudaFree(cuOutEdges);

    if (inverse_graph) {
        cudaFree(cuInOffset);
        cudaFree(cuInEdges);
    }
    if ((degree_options & IN_DEGREE) && (degree_options & OUT_DEGREE))
        cudaFree(cuInOutDegrees);
    else if (degree_options & IN_DEGREE)
        cudaFree(cuInDegrees);
    else if (degree_options & OUT_DEGREE)
        cudaFree(cuOutDegrees);
    __CUDA_ERROR("Graph Free");
}

template class cudaGraph<>;

} //@graph
