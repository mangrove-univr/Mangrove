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
#include <algorithm>
#include <exception>
#include <chrono>
#include <random>

#if __linux__
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

#include "Graph/Host/GraphSTD.hpp"
#include "Base/Host/fUtil.hpp"
#include "Base/Host/file_util.hpp"
#include "Base/Host/print_ext.hpp"

namespace graph {

template<typename node_t, typename edge_t, typename dist_t>
GraphSTD<node_t, edge_t, dist_t>::GraphSTD() :
                      GraphBase<node_t, edge_t>(),
                      OutOffset(nullptr), InOffset(nullptr),
                      OutEdges(nullptr), InEdges(nullptr),
                      OutDegrees(nullptr), InDegrees(nullptr),
                      COO_Edges(nullptr), coo_edges(0),
                      BFS(*this), SCC(*this) {}

template<typename node_t, typename edge_t, typename dist_t>
GraphSTD<node_t, edge_t, dist_t>::GraphSTD(EdgeType _edge_type) :
                        GraphBase<node_t, edge_t>(_edge_type),
                        OutOffset(nullptr), InOffset(nullptr),
                        OutEdges(nullptr), InEdges(nullptr),
                        OutDegrees(nullptr), InDegrees(nullptr),
                        COO_Edges(nullptr), coo_edges(0),
                        BFS(*this), SCC(*this) {}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::Allocate() {
    try {
        OutOffset = new edge_t[ V + 1 ];
        OutEdges = new node_t[ E ];
        OutDegrees = new node_t[ V ]();
        COO_Edges = new node_t2[ coo_edges ];
        if (Direction == EdgeType::UNDIRECTED) {
            InDegrees = OutDegrees;
            InOffset = OutOffset;
            InEdges = OutEdges;
            return;
        }
        InOffset = new edge_t[ V + 1 ];
        InEdges = new node_t[ E ];
        InDegrees = new degree_t[ V ]();
    }
    catch(std::bad_alloc& exc) {
        __ERROR("OUT OF MEMORY: Graph too Large !!   V: " << V << " E: " << E);
    }
}

template<typename node_t, typename edge_t, typename dist_t>
GraphSTD<node_t, edge_t, dist_t>::~GraphSTD() {
    if (OutOffset)   delete[] OutOffset;
    if (OutEdges)    delete[] OutEdges;
    if (OutDegrees)  delete[] OutDegrees;
    if (COO_Edges)   delete[] COO_Edges;
    if (Direction == EdgeType::UNDIRECTED)
        return;
    if (InOffset)    delete[] InOffset;
    if (InEdges)     delete[] InEdges;
    if (InDegrees)   delete[] InDegrees;
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::ToCSR(bool randomize) {
    if (randomize) {
        std::cout << std::endl << "Randomization..." << std::flush;
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        node_t* randomize_array = new node_t[V];
        std::iota(randomize_array, randomize_array + V, 0);
        std::shuffle(randomize_array, randomize_array+ V,
                     std::default_random_engine(seed));
        for (edge_t i = 0; i < coo_edges; i++) {
            COO_Edges[i][0] = randomize_array[ COO_Edges[i][0] ];
            COO_Edges[i][1] = randomize_array[ COO_Edges[i][1] ];
        }
        delete[] randomize_array;
    }
    std::cout << std::endl << "COO To CSR...\t" << std::flush;

    for (edge_t i = 0; i < coo_edges; i++) {
        const node_t source = COO_Edges[i][0];
        const node_t dest = COO_Edges[i][1];
        OutDegrees[source]++;
        if (Direction == EdgeType::UNDIRECTED)
            OutDegrees[dest]++;
        else if (Direction == EdgeType::DIRECTED)
            InDegrees[dest]++;
    }
    OutOffset[0] = 0;
    std::partial_sum(OutDegrees, OutDegrees + V, OutOffset + 1);

    degree_t* TMP = new degree_t[V]();
    for (edge_t i = 0; i < coo_edges; i++) {
        const node_t source = COO_Edges[i][0];
        const node_t dest = COO_Edges[i][1];
        OutEdges[ OutOffset[source] + TMP[source]++ ] = dest;
        if (Direction == EdgeType::UNDIRECTED)
            OutEdges[ OutOffset[dest] + TMP[dest]++ ] = source;
    }

    if (Direction == EdgeType::DIRECTED) {
        InOffset[0] = 0;
        std::partial_sum(InDegrees, InDegrees + V, InOffset + 1);

        std::fill(TMP, TMP + V, 0);
        for (edge_t i = 0; i < coo_edges; ++i) {
            const node_t dest = COO_Edges[i][1];
            InEdges[ InOffset[dest] + TMP[dest]++ ] = COO_Edges[i][0];
        }
    }
    delete[] TMP;
    std::cout << "Complete!" << std::endl << std::endl << std::flush;
}


#if __linux__

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::toBinary(const char* File) {
    const int fd = ::open(File, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

    const size_t file_size = (3 + (V + 1) * 2 + V * 2 + E * 2) * sizeof(int);
    std::cout << "Graph To binary file: " << File
              << " (" << (static_cast<float>(file_size) / (1 << 20)) << ") MB"
              << std::endl;

    ::lseek(fd, static_cast<long int>(file_size - 1), SEEK_SET);
    long int r = ::write(fd, "", 1);
    if (r != 1)
        __ERROR("write error");

    void* memory_mapped = ::mmap(0, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (memory_mapped == MAP_FAILED)
        __ERROR("memory_mapped error");
    ::madvise(memory_mapped, file_size, MADV_SEQUENTIAL);

    reinterpret_cast<node_t*>(memory_mapped)[0] = V;
    memory_mapped = (char*) memory_mapped + sizeof(node_t);
    reinterpret_cast<edge_t*>(memory_mapped)[0] = E;
    memory_mapped = (char*) memory_mapped + sizeof(edge_t);
    reinterpret_cast<EdgeType*>(memory_mapped)[0] = Direction;
    memory_mapped = (char*) memory_mapped + sizeof(EdgeType);

    xlib::Batch batch(file_size - sizeof(node_t) - sizeof(edge_t)
                                 - sizeof(EdgeType));
    batch.writeBinary(memory_mapped,
                      OutOffset, V + 1, InOffset, V + 1,
                      OutDegrees, V,  InDegrees, V,
                      OutEdges, E, InEdges, E);

    ::munmap(memory_mapped, file_size);
    ::close(fd);
}
#endif

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::print() const {
    using namespace xlib;
    printArray(OutOffset, V + 1, "OutOffset\t");
    printArray(OutEdges, E,      "OutEdges\t");
    printArray(OutDegrees, V,    "OutDegrees\t");
    if (Direction == EdgeType::UNDIRECTED)
        return;
    printArray(InOffset, V + 1, "InOffset\t");
    printArray(InEdges, E,      "InEdges\t\t");
    printArray(InDegrees, V,    "InDegrees\t");
}

} //@graph

#include "GraphSTD_Read.cpp"
#include "GraphSTD_BFS.cpp"
#include "GraphSTD_SCC.cpp"

template class graph::GraphSTD<>;
