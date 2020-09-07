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

#include <fstream>

namespace graph {

enum class      EdgeType { DIRECTED, UNDIRECTED, UNDEF_EDGE_TYPE };
enum class     GraphType { NORMAL, MULTIGRAPH, UNDEF_GRAPH_TYPE  };
enum class AttributeType { BINARY, INTEGER, REAL, SIGN           };

template<typename node_t = int, typename edge_t = int>
class GraphBase {
private:
    GraphBase(const GraphBase&) = delete;
    void operator=(const GraphBase&) = delete;

protected:
    virtual void Allocate() = 0;

    virtual void readMatrixMarket(std::ifstream& fin,
                                  bool randomize = false) = 0;

    virtual void readDimacs9     (std::ifstream& fin,
                                  bool randomize = false) = 0;

    virtual void readDimacs10    (std::ifstream& fin) = 0;

    virtual void readSnap        (std::ifstream& fin,
                                  bool randomize = 0) = 0;

    virtual void readKonect      (std::ifstream& fin,
                                  bool randomize = 0) = 0;

    virtual void readNetRepo     (std::ifstream& fin,
                                  bool randomize = 0) = 0;

    virtual void readBinary      (const char* File) = 0;

    virtual edge_t  getMatrixMarketHeader(std::ifstream& fin) final;
    virtual edge_t  getDimacs9Header     (std::ifstream& fin) final;
    virtual void    getDimacs10Header    (std::ifstream& fin) final;
    virtual void    getKonectHeader      (std::ifstream& fin) final;
    virtual void    getNetRepoHeader     (std::ifstream& fin) final;
    virtual edge_t  getSnapHeader        (std::ifstream& fin) final;

    virtual void printProperty() final;

public:
    node_t V;
    edge_t E;
    EdgeType Direction;

    GraphBase();
    virtual ~GraphBase();
    GraphBase(EdgeType _Direction);

    virtual void read(const char* File, bool randomize = false) final;
    virtual void print() const = 0;
};

} //@graph
