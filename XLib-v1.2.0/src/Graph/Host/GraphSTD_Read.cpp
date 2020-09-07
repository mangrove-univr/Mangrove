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
#include <sstream>
#include <cstring>  //std::strtok
#include "Base/Host/fUtil.hpp"
#include "Base/Host/file_util.hpp"

#if __linux__
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace graph {

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>
::readMatrixMarket(std::ifstream& fin, bool randomize) {
    coo_edges = GraphBase<node_t, edge_t>::getMatrixMarketHeader(fin);
    Allocate();
    xlib::Progress progress(static_cast<std::size_t>(coo_edges));

    for (int lines = 0; lines < coo_edges; lines++) {
        node_t index1, index2;
        fin >> index1 >> index2;

        COO_Edges[lines][0] = index1 - 1;
        COO_Edges[lines][1] = index2 - 1;

        progress.next(static_cast<std::size_t>(lines + 1));
        xlib::skipLines(fin);
    }
    ToCSR(randomize);
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::readDimacs9(std::ifstream& fin, bool randomize) {
    coo_edges = GraphBase<node_t, edge_t>::getDimacs9Header(fin);
    Allocate();
    xlib::Progress progress(static_cast<std::size_t>(coo_edges));

    int c;
    int lines = 0;
    std::string nil;
    while ((c = fin.peek()) != EOF) {
        if (c == static_cast<int>('a')) {
            node_t index1, index2;
            fin >> nil >> index1 >> index2;

            COO_Edges[lines][0] = index1 - 1;
            COO_Edges[lines][1] = index2 - 1;

            progress.next(static_cast<std::size_t>(++lines));
        }
        xlib::skipLines(fin);
    }
    ToCSR(randomize);
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::readKonect(std::ifstream& fin, bool randomize) {
    GraphBase<node_t, edge_t>::getKonectHeader(fin);
    xlib::UniqueMap<node_t> unique_map;
    std::vector<node_t> COO_Edges_v;
    int n_of_lines = 0;
    while (fin.good()) {
        node_t index1, index2;
        fin >> index1 >> index2;
        unique_map.insertValue(index1);
        unique_map.insertValue(index2);

        COO_Edges_v.push_back(index1 - 1);
        COO_Edges_v.push_back(index2 - 1);
        n_of_lines++;
    }
    V = static_cast<node_t>(unique_map.size());
    E = n_of_lines;
    coo_edges = n_of_lines;
    Allocate();
    std::copy(COO_Edges_v.begin(), COO_Edges_v.end(), (node_t*) COO_Edges);
    ToCSR(randomize);
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::readNetRepo(std::ifstream& fin, bool randomize) {
    GraphBase<node_t, edge_t>::getNetRepoHeader(fin);
    xlib::UniqueMap<node_t> unique_map;
    std::vector<node_t> COO_Edges_v;
    int n_of_lines = 0;
    while (!fin.eof()) {
        node_t index1, index2;
        fin >> index1;
        fin.ignore(1, ',');
        fin >> index2;
        unique_map.insertValue(index1);
        unique_map.insertValue(index2);

        COO_Edges_v.push_back(index1 - 1);
        COO_Edges_v.push_back(index2 - 1);
        n_of_lines++;
    }
    V = static_cast<node_t>(unique_map.size());
    E = n_of_lines;
    coo_edges = n_of_lines;
    Allocate();
    std::copy(COO_Edges_v.begin(), COO_Edges_v.end(), (node_t*) COO_Edges);
    ToCSR(randomize);
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::readDimacs10(std::ifstream& fin) {
    GraphBase<node_t, edge_t>::getDimacs10Header(fin);
    coo_edges = E;
    Allocate();
    xlib::Progress progress(static_cast<std::size_t>(coo_edges));

    OutOffset[0] = 0;
    int countEdges = 0;
    for (int lines = 0; lines < V; lines++) {
        std::string str;
        std::getline(fin, str);

        int degree = 0;
        char* token = std::strtok(const_cast<char*>(str.c_str()), " ");
        while (token != NULL) {
            degree++;
            node_t dest = std::stoi(token) - 1;
            OutEdges[countEdges] = dest;

            COO_Edges[countEdges][0] = lines;
            COO_Edges[countEdges][1] = dest;

            if (Direction == EdgeType::DIRECTED)
                InDegrees[dest]++;
            countEdges++;
            token = std::strtok(NULL, " ");
        }
        OutDegrees[lines] = degree;
        progress.next(static_cast<std::size_t>(lines + 1));
    }
    OutOffset[0] = 0;
    std::partial_sum(OutDegrees, OutDegrees + V, OutOffset + 1);

    if (Direction == EdgeType::DIRECTED) {
        InOffset[0] = 0;
        std::partial_sum(InDegrees, InDegrees + V, InOffset + 1);

        degree_t* TMP = new degree_t[V]();
        for (int i = 0; i < E; i++) {
            const node_t dest = COO_Edges[i][1];
            InEdges[ InOffset[dest] + TMP[dest]++ ] = COO_Edges[i][0];
        }
        delete[] TMP;
    }
    std::cout << std::endl << std::endl;
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::readSnap(std::ifstream& fin, bool randomize) {
    coo_edges = GraphBase<node_t, edge_t>::getSnapHeader(fin);
    Allocate();

    xlib::Progress progress(static_cast<std::size_t>(coo_edges));
    while (fin.peek() == '#')
        xlib::skipLines(fin);

    xlib::UniqueMap<node_t, node_t> Map;
    for (int lines = 0; lines < coo_edges; lines++) {
        node_t ID1, ID2;
        fin >> ID1 >> ID2;

        COO_Edges[lines][0] = Map.insertValue(ID1);
        COO_Edges[lines][1] = Map.insertValue(ID2);

        progress.next(static_cast<std::size_t>(lines + 1));
    }
    ToCSR(randomize);
}

#if __linux__

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::readBinary(const char* File) {
    const int fd = ::open(File, O_RDONLY, S_IRUSR);

    const size_t file_size = xlib::fileSize(File);
    void* memory_mapped = ::mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (memory_mapped == MAP_FAILED)
        __ERROR("memory_mapped error");
    ::madvise(memory_mapped, file_size, MADV_SEQUENTIAL);

    V = *((node_t*) memory_mapped);
    memory_mapped = (char*) memory_mapped + sizeof(node_t);
    E = *((edge_t*) memory_mapped);
    memory_mapped = (char*) memory_mapped + sizeof(edge_t);
    Direction = *((EdgeType*) memory_mapped);
    memory_mapped = (char*) memory_mapped + sizeof(EdgeType);
    Allocate();

    xlib::Batch batch(file_size - sizeof(node_t)
                       - sizeof(edge_t) - sizeof(EdgeType));
    batch.readBinary(memory_mapped,
                     OutOffset, V + 1, InOffset, V + 1,
                     OutDegrees, V,  InDegrees, V,
                     OutEdges, E, InEdges, E);

    ::munmap(memory_mapped, file_size);
    ::close(fd);
    std::cout << std::endl;
}
#endif

} //@graph
