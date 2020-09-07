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
#include "Base/Host/bitmask.hpp"

namespace graph {

template<typename node_t, typename edge_t, typename dist_t>
const dist_t GraphSTD<node_t, edge_t, dist_t>
::NOT_VISITED = std::numeric_limits<dist_t>::max();

template<typename node_t, typename edge_t, typename dist_t>
GraphSTD<node_t, edge_t, dist_t>::BFS_Class
    ::BFS_Class(const GraphSTD<node_t, edge_t, dist_t>& _graph) :
              graph(_graph), Eccentricity(nullptr), Queue(),
              Visited(), bfs_init(false), extern_bfs_init(false),
              Distance(nullptr) {}


template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::init() {
    extern_bfs_init = true;
    _init();
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::_init() {
    bfs_init = true;
    Queue.init(graph.V);
    Visited.init(graph.V);
    Distance = new dist_t[ graph.V ];
    reset();
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::close() {
    bfs_init = false;
    extern_bfs_init = false;
    Queue.free();
    Visited.free();
    delete[] Distance;
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::reset() {
    Queue.reset();
    Visited.reset();
    std::fill(Distance, Distance + graph.V, NOT_VISITED);
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::exec(node_t source) {
    if (!bfs_init)
        _init();

    Visited.set(source);
    Distance[source] = 0;
    Queue.insert(source);

    while (!Queue.isEmpty()) {
        const node_t next = Queue.extract();

        for (edge_t i = graph.OutOffset[next];
             i < graph.OutOffset[next + 1]; i++) {

            const node_t dest = graph.OutEdges[i];
            if (!Visited[dest]) {
                Visited.set(dest);
                Distance[dest] = Distance[next] + 1;
                Queue.insert(dest);
            }
        }
    }
    if (!extern_bfs_init)
        close();
}

template<typename node_t, typename edge_t, typename dist_t>
node_t GraphSTD<node_t, edge_t, dist_t>::BFS_Class::getVisitedNodes() const {
    return Queue.totalSize();
}

template<typename node_t, typename edge_t, typename dist_t>
edge_t GraphSTD<node_t, edge_t, dist_t>::BFS_Class::getVisitedEdges() const {
    if (Queue.totalSize() == graph.V)
        return graph.E;
    edge_t sum = 0;
    for (int i = 0; i < Queue.totalSize(); ++i)
        sum += graph.OutDegrees[ Queue.at(i) ];
    return sum;
}

template<typename node_t, typename edge_t, typename dist_t>
dist_t GraphSTD<node_t, edge_t, dist_t>::BFS_Class::getEccentricity() const {
    return Distance[ Queue.last() ];
}

//------------------------------------------------------------------------------

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>
::BFS_Class::statistics(std::vector<std::array<node_t, 4>>& Statistics,
                        node_t source) {
    if (!bfs_init)
        _init();
    const int PARENT = 0, PEER = 1, VALID = 2, NOT_VALID = 3;
    std::array<int, 4> counter = { { 0, 0, 0, 0 } };

    dist_t level = 0;
    Distance[source] = 0;
    Queue.insert(source);

    while (!Queue.isEmpty()) {
        const node_t node = Queue.extract();

        if (Distance[node] > level) {
            level++;
            Statistics.push_back(counter);
            counter[PARENT] = 0;
            counter[PEER] = 0;
            counter[VALID] = 0;
            counter[NOT_VALID] = 0;
        }

        for (edge_t i = graph.OutOffset[node];
            i < graph.OutOffset[node + 1]; i++) {

            const node_t dest = graph.OutEdges[i];
            if (Distance[dest] < level)
                counter[PARENT]++;
            else if (Distance[dest] == level)
                counter[PEER]++;
            else if (Distance[dest] == NOT_VISITED) {
                counter[VALID]++;
                Distance[dest] = level + 1;
                Queue.insert(dest);
            } else
                counter[NOT_VALID]++;
        }
    }
    if (!extern_bfs_init)
        close();
}

template<typename node_t, typename edge_t, typename dist_t>
std::vector<node_t> GraphSTD<node_t, edge_t, dist_t>::BFS_Class::WCC() {
    if (!bfs_init)
        _init();
    std::vector<node_t> wcc_distr;

    for (node_t source = 0; source < graph.V; source++) {
        if (Visited[source]) continue;
        node_t count = 0;
        Visited.set(source);
        Queue.insert(source);

        while (!Queue.isEmpty()) {
            const node_t next = Queue.extract();
            count++;

            for (edge_t i = graph.OutOffset[next];
                 i < graph.OutOffset[next + 1]; i++) {

                const node_t dest = graph.OutEdges[i];
                if (!Visited[dest]) {
                    Visited.set(dest);
                    Queue.insert(dest);
                }
            }
            for (edge_t i = graph.InOffset[next];
                i < graph.InOffset[next + 1]; i++) {

                const node_t incoming = graph.InEdges[i];
                if (!Visited[incoming]) {
                    Visited.set(incoming);
                    Queue.insert(incoming);
                }
            }
        }
        wcc_distr.push_back(count);
    }
    if (!extern_bfs_init)
        close();
    return wcc_distr;
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>
::BFS_Class::_eccentricity(node_t th_start, node_t th_end, int concurrency) {
    thread_local xlib::Queue<node_t> Queue_local(graph.V);
    thread_local xlib::Bitmask Visited_local(graph.V);
    thread_local dist_t* Distance_local = new dist_t[graph.V];

    for (node_t source = th_start; source < th_end; source += concurrency) {
        std::fill(Distance_local, Distance_local + graph.V, NOT_VISITED);
        Visited_local.reset();
        Distance_local[source] = 0;
        Visited_local.set(source);
        Queue_local.insert(source);

        while (!Queue_local.isEmpty()) {
            const node_t next = Queue_local.extract();

            for (edge_t i = graph.OutOffset[next];
                i < graph.OutOffset[next + 1]; i++) {

                const node_t dest = graph.OutEdges[i];
                if (!Visited_local[dest]) {
                    Visited_local.set(dest);
                    Distance_local[dest] = Distance_local[next] + 1;
                    Queue_local.insert(dest);
                }
            }
        }
        Eccentricity[source] = Distance_local[ Queue_local.last() ];
        Queue_local.reset();
    }
    delete[] Distance_local;
}

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::eccentricity() {
    Eccentricity = new dist_t[graph.V];
    const unsigned concurrency = std::thread::hardware_concurrency();
    std::thread threadArray[32];

    node_t chunk = graph.V / 100;
    for (int p = 0; p <= 100; p++) {
        node_t th_start = chunk * p;
        node_t th_end = std::min((p + 1) * chunk, graph.V);
        for (unsigned i = 0; i < concurrency; i++) {
            threadArray[i] = std::thread(
                            &GraphSTD<node_t, edge_t, dist_t>::
                                BFS_Class::_eccentricity,
                            this, th_start + i, th_end, concurrency);
        }
        for (unsigned i = 0; i < concurrency; i++)
            threadArray[i].join();
        std::cout << p << "%" << std::endl;
    }
}

//==============================================================================

template<typename node_t, typename edge_t, typename dist_t>
void GraphSTD<node_t, edge_t, dist_t>::BFS_Class::twoSweepDiameter() {
    /*int* Distance = new int[N];

    int lower_bound = 0, upper_bound;
    for (int i = 0; i < N; i++) {
        BFS_Init(Distance);
        BFS(rand_source);
        BFS_Init(Queue->last(), Distance);
        if (BFS_getEccentricity() > lower_bound)
            lower_bound = BFS_getEccentricity();

        if (UNDIRECTED) {
            BFS_Init(Distance);
            BFS(highDegree_source);
            if (BFS_getEccentricity() > lower_bound)
                lower_bound = BFS_getEccentricity();

            BFS_Init( Distance );
            BFS( Queue->last() );
            if (BFS_getEccentricity() < upper_bound)
                upper_bound = BFS_getEccentricity();

            if (lower_bound >= upper_bound)
                break;
        }
    }
    std::cout << lower_bound << std::endl;*/
}

} //@graph
