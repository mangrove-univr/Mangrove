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
#include <iterator>
#include "Base/Host/fUtil.hpp"
#include "Base/Host/file_util.hpp"
#include "Base/Host/print_ext.hpp"

#if __linux__
#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace graph {

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::readMatrixMarket(std::ifstream& fin, bool) {
    int n_of_lines = this->getMatrixMarketHeader(fin);
    this->Allocate();
    xlib::Progress progress(static_cast<std::size_t>(n_of_lines));

    for (int lines = 0; lines < n_of_lines; ++lines) {
        int index1, index2;
        fin >> index1 >> index2;
        xlib::skipLines(fin);
        index1--;
        index2--;

        OutDegrees[index1]++;
        if (this->Direction == EdgeType::UNDIRECTED)
            OutDegrees[index2]++;
        progress.next(static_cast<std::size_t>(lines + 1));
    }
}

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::readDimacs9(std::ifstream& fin, bool) {
    int n_of_lines = this->getDimacs9Header(fin);
    this->Allocate();
    xlib::Progress progress(static_cast<std::size_t>(n_of_lines));

    int lines = 0;
    std::string nil;
    do {
        int c = fin.peek();
        if (c == 'a') {
            int index1, index2;
            fin >> nil >> index1 >> index2;
            index1--;
            index2--;

            OutDegrees[index1]++;
            if (this->Direction == EdgeType::UNDIRECTED)
                OutDegrees[index2]++;
            lines++;
            progress.next(static_cast<std::size_t>(lines + 1));
        }
        xlib::skipLines(fin);
    } while (!fin.eof());
}

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::readDimacs10(std::ifstream& fin) {
    this->Allocate();
    xlib::Progress progress(static_cast<std::size_t>(V));

    while (fin.peek() == '%')
        xlib::skipLines(fin);
    xlib::skipLines(fin);

    for (int lines = 0; lines < V; lines++) {
        std::string str;
        std::getline(fin, str);

        std::istringstream stream(str);
        std::istream_iterator<std::string> iis(stream >> std::ws);

        OutDegrees[lines] = static_cast<degree_t>(
                      std::distance(iis, std::istream_iterator<std::string>()));
        progress.next(static_cast<std::size_t>(lines + 1));
    }
}

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::readSnap(std::ifstream& fin, bool) {
    int n_of_lines = this->getSnapHeader(fin);
    this->Allocate();
    xlib::Progress progress(static_cast<std::size_t>(n_of_lines));

    while (fin.peek() == '#')
        xlib::skipLines(fin);

    xlib::UniqueMap<int, int> Map;
    for (int lines = 0; lines < n_of_lines; lines++) {
        int ID1, ID2;
        fin >> ID1 >> ID2;

        int index1 = Map.insertValue(ID1);
        int index2 = Map.insertValue(ID2);

        OutDegrees[index1]++;
        if (this->Direction == EdgeType::UNDIRECTED)
            OutDegrees[index2]++;
        progress.next(static_cast<std::size_t>(lines + 1));
    }
}

#if __linux__

template<typename node_t, typename edge_t>
void GraphDegree<node_t, edge_t>::readBinary(const char* File) {
    const int fd = ::open(File, O_RDONLY, S_IRUSR);

    const size_t file_size = xlib::fileSize(File);
    void* memory_mapped = ::mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (memory_mapped == MAP_FAILED)
        __ERROR("memory_mapped error");
    ::madvise(memory_mapped, file_size, MADV_SEQUENTIAL);

    std::cout << "Binary file To Graph : " << File
              << " (" << (static_cast<float>(file_size) / (1 << 20)) << ") MB"
              << std::endl;

    int basic[3];
    xlib::Batch batch(file_size);
    batch.readBinary(memory_mapped,
                     basic, 3, OutDegrees, V);

    V = basic[0];
    E = basic[1];
    this->Direction = static_cast<EdgeType>(basic[2]);

    ::munmap(memory_mapped, file_size);
    ::close(fd);
}
#endif

} //@graph
