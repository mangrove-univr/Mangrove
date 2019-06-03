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
#pragma once
#include "../config.cuh"

namespace mangrove {

enum class READ_MODE         { STREAM, MMAP, GPU };

struct GenerateParamStr {
    std::vector<std::pair<double, double>> range_vector;
    int vars, sim_instants;
    bool overlapping, random;

    GenerateParamStr();
    GenerateParamStr(int _vars, int _sim_instants,
                     bool _overlapping, bool _random);
};

struct ReadParamStr {
    const char* trace_file;
    READ_MODE read_mode;
    bool overlapping, check_read;

    ReadParamStr();
    ReadParamStr(const char* _trace_file, READ_MODE _read_mode,
                 bool _overlapping, bool _check_read);

    ReadParamStr(const ReadParamStr& obj) = delete;
    ReadParamStr& operator=(const ReadParamStr& obj) = delete;
};

struct MiningParamStr {
    const char* output_file;
    char*  name [ MAX_VARS ] = {};
    bool print_inv;

    MiningParamStr();
    MiningParamStr(const char* _output_file, bool _print_inv);
    virtual ~MiningParamStr();

    MiningParamStr(const MiningParamStr& obj) = delete;
    MiningParamStr& operator=(const MiningParamStr& obj) = delete;
};

struct HostParamStr : MiningParamStr {
    bool multi;

    HostParamStr();
    HostParamStr(const char* _output_file, bool _multi, bool _print_inv);
    ~HostParamStr();

    HostParamStr(const HostParamStr &obj) = delete;
    HostParamStr& operator=(const HostParamStr &obj) = delete;
};

struct GPUParamStr : MiningParamStr {
    bool check_result;

    GPUParamStr();
    GPUParamStr(const char* _output_file, bool _check_result, bool _print_inv);
    ~GPUParamStr();

    GPUParamStr(const GPUParamStr& obj) = delete;
    GPUParamStr& operator=(const GPUParamStr& obj) = delete;
};

} //@mangrove
