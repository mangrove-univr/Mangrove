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

#include "Graph/Host/GraphSTD.hpp"

namespace graph {

const unsigned OUT_DEGREE = 1;
const unsigned  IN_DEGREE = 2;

using _node_t = int;
using _cunode_t = int2;
using _cuedge_t = int;
using _degree_t = _node_t;
using _degree2_t = _cunode_t;

extern __constant__ _node_t devV;
extern __constant__ _cuedge_t devE;
extern __constant__ _cunode_t* devOutOffset;
extern __constant__ _cuedge_t*  devOutEdges;
extern __constant__ _cunode_t* devInOffset;
extern __constant__ _cuedge_t* devInEdges;
extern __constant__ _degree2_t* devInOutDegrees;
extern __constant__ _degree_t* devOutDegrees;
extern __constant__ _degree_t* devInDegrees;

template<typename node_t = int, typename edge_t = int, typename cunode_t = int2,
         typename dist_t = typename std::make_unsigned<node_t>::type>
class cudaGraph {

static_assert(std::is_unsigned<dist_t>::value, "dist_t must be unsigned");
using degree_t = node_t;
using degree2_t = cunode_t;

static_assert(std::is_same<node_t, _node_t>::value,
              PRINT_ERR("wrong type"));
static_assert(std::is_same<_cuedge_t, _cuedge_t>::value,
              PRINT_ERR("wrong type"));
static_assert(std::is_same<cunode_t, _cunode_t>::value,
              PRINT_ERR("wrong type"));

protected:
    cunode_t* cuOutOffset;
    edge_t* cuOutEdges;
    cunode_t* cuInOffset;
    edge_t* cuInEdges;

    degree_t* cuOutDegrees;
    degree_t* cuInDegrees;
    degree2_t* cuInOutDegrees;

    GraphSTD<node_t, edge_t, dist_t>& graph;
    unsigned degree_options;
    node_t V;
    edge_t E;
    bool inverse_graph;

public:
    cudaGraph(GraphSTD<node_t, edge_t, dist_t>& graph,
              bool _inverse_graph = true,
              unsigned _degree_options = 0x0);

    virtual ~cudaGraph();
};

} //@graph
