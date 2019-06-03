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
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include "XLib.hpp"

#include "Utility.hpp"

using namespace xlib;

namespace mangrove_support {

ParameterStr::ParameterStr() : var_file(nullptr),
                               host(false), gpu(false), gen(false),
                               mining_type(MINING_TYPE::VOID) {}

void setParameters(GenerateParamStr& GenerateParam, ReadParamStr& ReadParam,
                   HostParamStr& HostParam, GPUParamStr& GPUParam,
                   ParameterStr& Parameter,
                   int argc, char* argv[]) {

    if (argc < 2)
        __ERROR("Wrong paramenters : " << argv[0]
                << " --help for Mangrove Usage");
    for (int i = 1; i < argc; i++) {
        std::string param(argv[i]);
        if (param.compare("-G") == 0 && i + 2 < argc &&
            isDigit(argv[i + 1]) && isDigit(argv[i + 2])) {

            GenerateParam.vars = std::stoi(argv[++i]);
            GenerateParam.sim_instants = std::stoi(argv[++i]);
            if (GenerateParam.vars < 3)
                  __ERROR("Wrong paramenters B")

            if (i + 1 < argc &&
                std::string(argv[i + 1]).compare("-random") == 0) {

                unsigned seed = std::chrono::system_clock::now()
                                .time_since_epoch().count();
                std::default_random_engine generator(seed);
                std::uniform_real_distribution<> distribution(
                                       std::numeric_limits<numeric_t>::lowest(),
                                       std::numeric_limits<numeric_t>::max());

                i++;
                GenerateParam.random = true;
                for (int j = 0; j < GenerateParam.vars; j++) {
                    GenerateParam.range_vector.push_back(
                        std::pair<double, double>(distribution(generator),
                                                  distribution(generator)));
                }
            }
            else if (i + 2 < argc &&
                     std::string(argv[i + 1]).compare("-range") == 0) {

                std::string ranges(argv[i + 2]);
                GenerateParam.random = true;

                bool flag = true;
                unsigned init = 0;
                unsigned pos = ranges.find(",");
                while (true) {
                    std::string single_range(ranges.substr(init, pos - init));
                    std::cout << single_range << std::endl;

                    auto r1 = std::stof(std::strtok(const_cast<char*>(single_range.c_str()), "_"));
                    auto r2 = std::stof(std::strtok(nullptr, "_"));
                    GenerateParam.range_vector.push_back(
                        std::pair<double, double>(r1, r2));

                    init = pos + 1;
                    pos = ranges.find(",", init);
                    if (!flag)   break;
                    if (pos >= ranges.size())
                        flag = false;
                }
                i += 2;
            }
            else {
                for (int j = 0; j < GenerateParam.vars; j++) {
                    GenerateParam.range_vector.push_back(
                        std::pair<double, double>(1.0, 1.0));
                }
            }
            Parameter.gen = true;
        }
        else if (param.compare("-T") == 0 && i + 1 < argc) {
            ReadParam.trace_file = argv[++i];
            Parameter.gen = false;
        }
        else if (param.compare("-S") == 0)
            Parameter.host = true;
        else if (param.compare("-M") == 0) {
            HostParam.multi = true;
            Parameter.host = true;
        }
        else if (param.compare("-overlap") == 0) {
            ReadParam.overlapping = true;
            GenerateParam.overlapping = true;
        }
        else if (param.compare("-check-read") == 0)
            ReadParam.check_read = true;
        else if (param.compare("-varfile") == 0 && i + 1 < argc) {
            Parameter.var_file = argv[++i];
            xlib::checkRegularFile(Parameter.var_file);
        }
        else if (param.compare("-GPU") == 0)
            Parameter.gpu = true;
        else if (param.compare("-check-results") == 0)
            GPUParam.check_result = true;
        else if (param.compare("-mining=bool") == 0)
            Parameter.mining_type = MINING_TYPE::BOOL;
        else if (param.compare("-mining=numeric") == 0)
            Parameter.mining_type = MINING_TYPE::NUMERIC;
        else if (param.compare("-read=MMAP") == 0)
            ReadParam.read_mode = READ_MODE::MMAP;
        else if (param.compare("-read=GPU") == 0)
            ReadParam.read_mode = READ_MODE::GPU;
        else if (param.compare("--help") == 0) {
            std::ifstream SyntaxtraceFile("../Syntax.txt");
            std::cout << SyntaxtraceFile.rdbuf();
            std::exit(EXIT_SUCCESS);
        }
        else if (param.compare("-output") == 0 && i + 1 < argc) {
            HostParam.output_file = argv[++i];
            GPUParam.output_file = HostParam.output_file;
        }
        else
           __ERROR("Wrong parameter : " << argv[i] << std::endl)
    }
    if (Parameter.mining_type == MINING_TYPE::VOID)
        __ERROR("Missing parameter : -mining=<mining_type>")
    if (ReadParam.trace_file == nullptr && GenerateParam.vars == -1) {
         __ERROR("Wrong parameter : missing trace")
    }
}

void getVariables(const char* fileName, int vars, MiningParamStr& mining_param){
    if (fileName != nullptr) {
        std::ifstream var_file(fileName);
        xlib::checkRegularFile(fileName);

        int index = 0;
        while ( !var_file.eof() ) {
            std::string var_name;
            var_file >> var_name;
            if (var_name.empty()) continue;
            mining_param.name[index] = new char[var_name.length() + 1]();
            var_name.copy(mining_param.name[index++], var_name.length());
        }
        var_file.close();

        if (vars != index) {
            __ERROR("The number of variables in varfile : " << fileName <<
                    " and trace does not match" << std::endl);
        }
    }
    else {
        for (int i = 0; i < vars; i++) {
            std::string s = std::string("randomVar_") + std::to_string(i);
            mining_param.name[i] = new char[s.length() + 1]();
            s.copy(mining_param.name[i], s.length());
        }
    }
}

} //@mangrove_support
