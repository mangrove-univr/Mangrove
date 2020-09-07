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
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Graph/Host/GraphBase.hpp"
#include "Base/Host/file_util.hpp"
#include "Base/Host/fUtil.hpp"

namespace graph {

template<typename node_t, typename edge_t>
GraphBase<node_t, edge_t>::GraphBase() :
                             V(0), E(0), Direction(EdgeType::UNDEF_EDGE_TYPE) {}

template<typename node_t, typename edge_t>
GraphBase<node_t, edge_t>::GraphBase(EdgeType _Direction) :
                             V(0), E(0), Direction(_Direction) {}

template<typename node_t, typename edge_t>
GraphBase<node_t, edge_t>::~GraphBase() {}

template<typename node_t, typename edge_t>
void GraphBase<node_t, edge_t>::read(const char* File, bool randomize) {
    xlib::checkRegularFile(File);
    size_t size = xlib::fileSize(File);

    xlib::ThousandSep TS;

    std::cout << std::endl << "Reading Graph File...\t"
              << xlib::extractFileName(File) << "\tSize: "
              <<  size / (1024 * 1024) << " MB" << std::flush;

    std::string file_ext = xlib::extractFileExtension(File);
    if (file_ext.compare(".bin") == 0) {
        std::cout << "            (Binary)   " << std::endl << std::flush;
        if (randomize) {
            std::cerr << "#input randomization ignored on binary format"
                      << std::endl;
        }
        this->readBinary(File);
        this->printProperty();
        return;
    }

    std::ifstream fin(File);
    std::string s;
    fin >> s;
    fin.seekg(std::ios::beg);

    if (file_ext.compare(".mtx") == 0 && s.compare("%%MatrixMarket") == 0) {
        std::cout << "     (MatrixMarket)" << std::endl << std::flush;
        this->readMatrixMarket(fin, randomize);
    }
    else if (file_ext.compare(".graph") == 0 && xlib::isDigit(s)) {
        std::cout << "       (Dimacs10th)" << std::endl << std::flush;
        if (randomize) {
            std::cerr << "#input randomization ignored on Dimacs10th format"
                      << std::endl;
        }
        this->readDimacs10(fin);
    }
    else if (file_ext.compare(".gr") == 0
               && (s.compare("c") == 0 || s.compare("p") == 0)) {
        std::cout << "        (Dimacs9th)" << std::endl << std::flush;
        this->readDimacs9(fin, randomize);
    }
    else if (file_ext.compare(".txt") == 0 && s.compare("#") == 0) {
        std::cout << "             (SNAP)" << std::endl << std::flush;
        this->readSnap(fin, randomize);
    }
    else if (file_ext.compare(".edges") == 0) {
        std::cout << "   (Net Repository)" << std::endl << std::flush;
        this->readNetRepo(fin, randomize);
    }
    else if (s.compare("%") == 0) {
       std::cout << "        (Konect)" << std::endl << std::flush;
       this->readKonect(fin, randomize);
    } else
        __ERROR("Graph type not recognized");

    fin.close();
    this->printProperty();
}

template<typename node_t, typename edge_t>
void GraphBase<node_t, edge_t>::printProperty() {
    xlib::ThousandSep TS;
    std::string graphDir = Direction == EdgeType::UNDIRECTED
                                ? "\tGraphType: Undirected"
                                : "\tGraphType: Directed";
    std::cout << "Nodes: " << V << "\tEdges: " << E
              << graphDir << "\tDegree AVG: " << std::fixed
              << std::setprecision(1) << static_cast<double>(E) / V
              << std::endl;
}

//==============================================================================

template<typename node_t, typename edge_t>
edge_t GraphBase<node_t, edge_t>
::getMatrixMarketHeader(std::ifstream& fin) {
    std::string header_lines;
    std::getline(fin, header_lines);
    if (Direction == EdgeType::UNDEF_EDGE_TYPE) {
       Direction = header_lines.find("symmetric") != std::string::npos ?
                             EdgeType::UNDIRECTED : EdgeType::DIRECTED;
    }
    while (fin.peek() == '%')
        xlib::skipLines(fin);

    std::size_t n_of_lines;
    fin >> V >> header_lines >> n_of_lines;
    if (n_of_lines > std::numeric_limits<edge_t>::max())
        __ERROR("n_of_lines > max value of <edge_t>")
    E = (Direction == EdgeType::UNDIRECTED) ?
                        static_cast<edge_t>(n_of_lines) * 2 :
                        static_cast<edge_t>(n_of_lines);
    xlib::skipLines(fin);
    return static_cast<edge_t>(n_of_lines);
}
/*if (header_lines.find("real") != std::string::npos)
    FileAttributeType = AttributeType::REAL;
else if (header_lines.find("integer") != std::string::npos)
    FileAttributeType = AttributeType::INTEGER;
else
    FileAttributeType = AttributeType::BINARY;*/

template<typename node_t, typename edge_t>
edge_t GraphBase<node_t, edge_t>::getDimacs9Header(std::ifstream& fin) {
    while (fin.peek() == 'c')
        xlib::skipLines(fin);

    std::size_t n_of_lines;
    std::string nil;
    fin >> nil >> nil >> V >> n_of_lines;
    if (n_of_lines > std::numeric_limits<edge_t>::max())
        __ERROR("n_of_lines > max value of <edge_t>")
    Direction = EdgeType::DIRECTED;
    E = static_cast<edge_t>(n_of_lines);
    return static_cast<edge_t>(n_of_lines);;
    //FileAttributeType = AttributeType::INTEGER;
}

template<typename node_t, typename edge_t>
void GraphBase<node_t, edge_t>::getDimacs10Header(std::ifstream& fin) {
    while (fin.peek() == '%')
        xlib::skipLines(fin);

    std::size_t n_of_lines;
    fin >> V >> n_of_lines;
    if (n_of_lines > std::numeric_limits<edge_t>::max())
        __ERROR("n_of_lines > max value of <edge_t>")

    if (fin.peek() != '\n') {
        std::string str;
        fin >> str;
        if (str.compare("100") == 0)
            Direction = EdgeType::DIRECTED;
        else
            __ERROR("graph read error : wrong format");
    } else
        Direction = EdgeType::UNDIRECTED;

    xlib::skipLines(fin);

    E = (Direction == EdgeType::UNDIRECTED) ?
                        static_cast<edge_t>(n_of_lines) * 2 :
                        static_cast<edge_t>(n_of_lines);
    //FileAttributeType = AttributeType::BINARY;
}

template<typename node_t, typename edge_t>
void GraphBase<node_t, edge_t>::getKonectHeader(std::ifstream& fin) {
    std::string str;
    fin >> str >> str;
    if (Direction == EdgeType::UNDEF_EDGE_TYPE) {
        Direction = str.compare("asym") == 0 ? EdgeType::DIRECTED
                                             : EdgeType::UNDIRECTED;
    }
    xlib::skipLines(fin);
    //FileAttributeType = AttributeType::BINARY;
}

template<typename node_t, typename edge_t>
void GraphBase<node_t, edge_t>::getNetRepoHeader(std::ifstream& fin) {
    std::string str;
    fin >> str >> str;
    if (Direction == EdgeType::UNDEF_EDGE_TYPE) {
        Direction = str.compare("directed") == 0 ? EdgeType::DIRECTED
                                                 : EdgeType::UNDIRECTED;
    }
    xlib::skipLines(fin);
    //FileAttributeType = AttributeType::BINARY;
}

template<typename node_t, typename edge_t>
edge_t GraphBase<node_t, edge_t>::getSnapHeader(std::ifstream& fin) {
    std::string tmp;
    fin >> tmp >> tmp;
    if (Direction == EdgeType::UNDEF_EDGE_TYPE) {
           Direction = tmp.compare("Undirected") == 0 ? EdgeType::UNDIRECTED
                                                      : EdgeType::DIRECTED;
    }
    xlib::skipLines(fin);

    std::size_t n_of_lines = 0;
    while (fin.peek() == '#') {
        std::getline(fin, tmp);
        if (tmp.substr(2, 6).compare("Nodes:") == 0) {
            std::istringstream stream(tmp);
            stream >> tmp >> tmp >> V >> tmp >> n_of_lines;
            break;
        }
    }
    if (n_of_lines > std::numeric_limits<edge_t>::max())
        __ERROR("n_of_lines > max value of <edge_t>")

    xlib::skipLines(fin);
    E = (Direction == EdgeType::UNDIRECTED) ?
                        static_cast<edge_t>(n_of_lines) * 2 :
                        static_cast<edge_t>(n_of_lines);
    return static_cast<edge_t>(n_of_lines);
    /*std::string header_lines;
    std::getline(fin, header_lines);
    FileAttributeType = header_lines.find("Sign") != std::string::npos ?
                            AttributeType::SIGN :: AttributeType::BINARY;*/
}

template class GraphBase<int, int>;

} //@graph
