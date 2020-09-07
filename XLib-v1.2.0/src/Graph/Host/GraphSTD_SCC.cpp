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
#include "Base/Host/fUtil.hpp"

namespace graph {

template<typename node_t, typename edge_t, typename dist_t>
const node_t GraphSTD<node_t, edge_t, dist_t>::SCC_Class::INDEX_UNDEF = -1;

template<typename node_t, typename edge_t, typename dist_t>
GraphSTD<node_t, edge_t, dist_t>::SCC_Class
    ::SCC_Class(const GraphSTD<node_t, edge_t, dist_t>& _graph) :
            graph(_graph), LowLink(nullptr), Index(nullptr), SCC_set(nullptr),
            InStack(0), Queue(0), scc_init(false), extern_scc_init(false),
            curr_index(0), SCC_index(0) {}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::SCC_Class::_init() {
    scc_init = true;
    LowLink = new node_t[graph.V];
    Index = new node_t[graph.V];
    SCC_set = new node_t[graph.V];
    InStack.init(graph.V);
    Queue.init(graph.V);

    curr_index = 0;
    SCC_index = 0;
    std::fill(Index, Index + graph.V, SCC_Class::INDEX_UNDEF);
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::SCC_Class::init() {
    extern_scc_init = true;
    _init();
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::SCC_Class::close() {
    scc_init = false;
    extern_scc_init = false;
    delete[] LowLink;
    delete[] Index;
    delete[] SCC_set;
    InStack.free();
    Queue.free();
}

template<typename node_t, typename edge_t, typename dist_t>
std::vector<node_t> GraphSTD<node_t, edge_t, dist_t>::SCC_Class::exec() {
    xlib::stackManagement StackSTR;
    StackSTR.checkUnlimited();

    if (!scc_init)
        _init();
    std::vector<node_t> scc_distr;
    for (node_t i = 0; i < graph.V; i++) {
        if (Index[i] == -1) {
            singleSCC(i, &scc_distr);
            Queue.reset();
        }
    }
    if (!extern_scc_init)
        close();
    return scc_distr;
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::SCC_Class
::singleSCC(node_t source, std::vector<node_t>* scc_distr_v) {
    Queue.insert(source);
    Index[source] = LowLink[source] = curr_index++;
    InStack.set(source);

    for (edge_t i = graph.OutOffset[source];
        i < graph.OutOffset[source + 1]; i++) {

        const node_t dest = graph.OutEdges[i];
        if ( Index[dest] == INDEX_UNDEF ) {
            singleSCC(dest, scc_distr_v);
            LowLink[source] = std::min(LowLink[source], LowLink[dest]);
        } else if ( InStack[dest] )
            LowLink[source] = std::min(LowLink[source], Index[dest]);
    }

    if (Index[source] == LowLink[source]) {
        node_t extracted;
        node_t SCC_size = 0;
        do {
            SCC_size++;
            extracted = Queue.template extract<xlib::QueuePolicy::LIFO>();
            InStack.unset(extracted);
            SCC_set[extracted] = SCC_index;
        } while (extracted != source);
        scc_distr_v->push_back(SCC_size);
        SCC_index++;
    }
}

} //@graph
