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

#include <vector>
#include "Base/Host/fUtil.hpp"
#include "GraphDegree.hpp"
#include "Base/Host/fast_queue.hpp"
#include "Base/Host/bitmask.hpp"

namespace graph {

template<typename node_t = int, typename edge_t = int,
         typename dist_t = typename std::make_unsigned<node_t>::type>
class GraphSTD : public GraphBase<node_t, edge_t> {
    static_assert(std::is_unsigned<dist_t>::value, "dist_t must be unsigned");
    using degree_t = node_t;
    using node_t2  = node_t[2];

private:
    static const dist_t NOT_VISITED;

    GraphSTD(const GraphSTD&) = delete;
    void operator=(const GraphSTD&) = delete;

    void Allocate() override;

    void readMatrixMarket(std::ifstream& fin, bool randomize) override;
    void readDimacs9     (std::ifstream& fin, bool randomize) override;
    void readDimacs10    (std::ifstream& fin) override;
    void readSnap        (std::ifstream& fin, bool randomize) override;
    void readKonect      (std::ifstream& fin, bool randomize) override;
    void readNetRepo     (std::ifstream& fin, bool randomize) override;
    void readBinary      (const char* File)   override;

    void ToCSR(bool random);
    //void ToCOO();

    struct BFS_Class {
    private:
        BFS_Class(const BFS_Class&) = delete;
        void operator=(const BFS_Class&) = delete;
        const GraphSTD<node_t, edge_t, dist_t>& graph;

        dist_t* Eccentricity;
        xlib::Queue<node_t> Queue;
        xlib::Bitmask Visited;
        bool bfs_init, extern_bfs_init;

        void _init();
        void _close();
        void _eccentricity(node_t s_index, node_t e_index, int concurrency);
    public:
        BFS_Class(const GraphSTD<node_t, edge_t, dist_t>& _graph);

        dist_t* Distance;

        void exec(node_t source);
        void init();
        void close();
        void reset();

        node_t  getVisitedNodes()    const;
        edge_t  getVisitedEdges()    const;
        dist_t  getEccentricity()    const;
        void    statistics(std::vector<std::array<node_t, 4>>& Statistics,
                           node_t source);
        std::vector<node_t> WCC();
        void    twoSweepDiameter();
        void    eccentricity();
    };

    struct SCC_Class {
    private:
        SCC_Class(const SCC_Class&) = delete;
        void operator=(const SCC_Class&) = delete;
        const GraphSTD<node_t, edge_t, dist_t>& graph;

        static const node_t INDEX_UNDEF;

        node_t* LowLink;
        node_t* Index;
        int* SCC_set;
        xlib::Bitmask InStack;
        xlib::Queue<node_t> Queue;

        bool scc_init, extern_scc_init;
        node_t curr_index; int SCC_index;

        void _init();
        void singleSCC(node_t source, std::vector<node_t>* scc_distr_v);
    public:
        SCC_Class(const GraphSTD<node_t, edge_t, dist_t>& _graph);

        void init();
        void close();
        std::vector<node_t> exec();
    };

public:
    using GraphBase<node_t, edge_t>::V;
    using GraphBase<node_t, edge_t>::E;
    using GraphBase<node_t, edge_t>::Direction;

    edge_t *OutOffset, *InOffset;
    node_t *OutEdges, *InEdges;
    degree_t *OutDegrees, *InDegrees;
    node_t2* COO_Edges;
    edge_t coo_edges;

    GraphSTD();
    GraphSTD(const EdgeType _edgeType);
    virtual ~GraphSTD();

    void print() const override;

#if defined(__linux__)
    virtual void toBinary( const char* File );
#endif

    BFS_Class BFS;
    SCC_Class SCC;
};

} //@graph
