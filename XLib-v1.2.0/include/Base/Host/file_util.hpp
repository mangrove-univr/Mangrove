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
#include <iostream>

namespace xlib {

class Progress {
private:
    std::size_t progressC, total;
    double fchunk;
    std::size_t nextChunk;
public:
    Progress(std::size_t total);
    void next(std::size_t progress);
    void perCent(std::size_t progress);
};

struct Batch {
    Progress progress;
    std::size_t partial;

    Batch(std::size_t total);

    template<typename = void, typename = void, typename... Ts>
    void writeBinary(void* pointer);

    template<typename T1, typename T2, typename... Ts>
    void writeBinary(void* pointer, T1* Data, T2 size, Ts... args);

    template<typename = void, typename = void, typename... Ts>
    void readBinary(void* pointer);

    template<typename T1, typename T2, typename... Ts>
    void readBinary(void* pointer, T1* Data, T2 size, Ts... args);
};

void        checkRegularFile(const char* File);
void        checkRegularFile(std::ifstream& fin);
std::size_t fileSize(const char* File);
std::size_t fileSize(std::ifstream& fin);
std::string extractFileName(std::string s);
std::string extractFileNameExtension(std::string s);
std::string extractFileExtension(std::string str);
std::string extractFilePath(std::string str);
std::string extractFilePathNoExtension(std::string str);
void        skipLines(std::istream& fin, const int nof_lines = 1);

} //@xlib

#include "impl/file_util.i.hpp"
