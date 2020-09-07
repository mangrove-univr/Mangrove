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
#include <map>
#include "GraphSTD.hpp"
#include "Base/Base.hpp"

namespace graph {

class GraphWeight : public GraphSTD {
private:
    void readMatrixMarket(std::ifstream& fin, const int nof_lines);
    void readDimacs9(std::ifstream& fin,const int nof_lines);
    void readDimacs10(std::ifstream& fin);
    void readSnap(std::ifstream& fin, const int nof_lines);
    void readBinary(const char* File);
    GraphWeight(const GraphWeight&);
    void operator=(const GraphWeight&);

    void ToCSR();
    bool dimacs9;

public:
    weight_t* Weights;
    GraphWeight(const int _V, const int _E, const EdgeType _edgeType);
    virtual ~GraphWeight();

    void read(const char* File, const int nof_lines);

    void toBinary( const char* File );
};

} //@graph
