 /*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

Mangrove is provided under the terms of The MIT License (MIT):

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
#include "DataTypes/ModuleSTR.hpp"

namespace mangrove {

GenerateParamStr::GenerateParamStr() : vars(0), sim_instants(0),
                                       overlapping(false), random(false) {}

GenerateParamStr::GenerateParamStr(int _vars, int _sim_instants,
                                   bool _overlapping, bool _random) :
                                   vars(_vars), sim_instants(_sim_instants),
                                   overlapping(_overlapping), random(_random) {}


ReadParamStr::ReadParamStr() : trace_file(nullptr),
                               read_mode(READ_MODE::STREAM),
                               overlapping(false), check_read(false) {}

ReadParamStr::ReadParamStr(const char* _trace_file, READ_MODE _read_mode,
                           bool _overlapping, bool _check_read) :
                           trace_file(_trace_file), read_mode(_read_mode),
                           overlapping(_overlapping),  check_read(_check_read){}


MiningParamStr::MiningParamStr() : output_file(nullptr),
                                   print_inv(false)  {}

MiningParamStr::MiningParamStr(const char* _output_file, bool _print_inv) :
                               output_file(_output_file),
                               print_inv(_print_inv) {}

MiningParamStr::~MiningParamStr() {
    int i = 0;
    while (name[i])
        delete[] name[i++];
}


HostParamStr::HostParamStr() :  MiningParamStr(), multi(false) {}

HostParamStr::HostParamStr(const char* _output_file, bool _print_inv,
                           bool _multi) :
                           MiningParamStr(_output_file, _print_inv),
                           multi(_multi) {}

HostParamStr::~HostParamStr() {}


GPUParamStr::GPUParamStr() : MiningParamStr(), check_result(false) {}

GPUParamStr::GPUParamStr(const char* _output_file,
                         bool _check_result, bool _print_inv) :
                         MiningParamStr(_output_file, _print_inv),
                         check_result(_check_result) {}

GPUParamStr::~GPUParamStr() {}

} //@mangrove
