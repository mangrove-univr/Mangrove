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
#include <exception>
#include "Graph/Host/GraphDegree.hpp"
#include "Base/Host/fUtil.hpp"
#include "Base/Host/print_ext.hpp"

namespace graph {

template<typename node_t, typename edge_t>
GraphDegree<node_t, edge_t>::GraphDegree() :
                   GraphBase<node_t, edge_t>(), OutDegrees(nullptr) {}

template<typename node_t, typename edge_t>
GraphDegree<node_t, edge_t>::GraphDegree(EdgeType _edge_type) :
                   GraphBase<node_t, edge_t>(_edge_type), OutDegrees(nullptr) {}

template<typename node_t, typename edge_t>
GraphDegree<node_t, edge_t>::~GraphDegree() {
    if (OutDegrees)
        delete[] OutDegrees;
}

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::Allocate() {
    this->printProperty();
    try {
        OutDegrees = new degree_t[ V ]();
    } catch(std::bad_alloc& exc) {
        __ERROR("OUT OF MEMORY: Graph too Large !!");
    }
}

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::print() const {
    xlib::printArray(OutDegrees, V, "OutDegrees\t");
}

} //@graph

#include "GraphDegree_Read.cpp"

template class graph::GraphDegree<int, int>;
