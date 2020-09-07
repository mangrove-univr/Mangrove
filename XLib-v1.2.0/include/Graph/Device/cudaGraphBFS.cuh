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
#pragma once

#include "cudaGraph.cuh"

namespace graph {

using _dist_t = typename std::make_unsigned<_node_t>::type;

extern __constant__ _node_t* devF1;
extern __constant__ _node_t* devF2;
extern __constant__ _dist_t* devDistance;

extern __device__ int devF2_size[4];

template<typename node_t = int, typename edge_t = int, typename cunode_t = int2,
         typename dist_t = typename std::make_unsigned<node_t>::type>
class cudaGraphBFS : public cudaGraph<node_t, edge_t, cunode_t, dist_t> {
public:
    static_assert(std::is_unsigned<dist_t>::value, "dist_t must be unsigned");
    using cudaGraph<node_t, edge_t, cunode_t>::V;
    using cudaGraph<node_t, edge_t, cunode_t>::E;

    static const dist_t INF_DIST;

    node_t* cuF1, *cuF2;
    dist_t* cuDistance;
    std::size_t max_frontier_nodes;

    cudaGraphBFS(GraphSTD<node_t, edge_t, dist_t>& graph,
                 bool _inverse_graph = true,
                 unsigned _degree_options = 0x0);

    virtual ~cudaGraphBFS();

    virtual void reset(int* Sources, int n_of_sources, bool update_distance);

    void AllocateFrontiers();
};

} //@graph
