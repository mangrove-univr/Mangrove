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
#include "Base/Host/file_util.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include "Base/Host/numeric.hpp"

namespace xlib {

void skipLines(std::istream& fin, const int nof_lines) {
     for (int i = 0; i < nof_lines; ++i)
         fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::size_t fileSize(const char* File) {
      std::ifstream fin(File);
     return fileSize(fin);
}

std::size_t fileSize(std::ifstream& fin) {
    fin.seekg (0L, std::ios::beg);
     std::iostream::pos_type startPos = fin.tellg();
     fin.seekg (0L, std::ios::end);
     std::iostream::pos_type endPos = fin.tellg();
     fin.close();
     return static_cast<std::size_t>(endPos - startPos);
}

void checkRegularFile(std::ifstream& fin) {
    if (!fin.is_open() || fin.fail() || fin.bad() || fin.eof())
        throw std::ios_base::failure("Unable to read file");
    try {
        char c;    fin >> c;
    } catch (std::ios_base::failure& e) {
        throw std::ios_base::failure("Unable to read file");
    }
    fin.seekg(0, std::ios::beg);
}

void checkRegularFile(const char* File) {
    std::ifstream fin(File);
    try {
        checkRegularFile(fin);
    } catch (std::ios_base::failure& e) {
        throw std::ios_base::failure(
                             std::string("Unable to read file : ") + File);
    }
    fin.close();
}

std::string extractFilePath(std::string str) {
    return str.substr(0, str.find_last_of("/") + 1);
}

std::string extractFilePathNoExtension(std::string str) {
    return str.substr(0, str.find_last_of("."));
}

std::string extractFileName(std::string str) {
     std::string name2 = str.substr(0, str.find_last_of("."));

     const long unsigned found = name2.find_last_of("/");
     if (found != std::string::npos)
         return name2.substr(found + 1);
     return name2;
}

std::string extractFileExtension(std::string str) {
    return str.substr(str.find_last_of("."));
}

// ----------------------------------- PROGRESS --------------------------------

Progress::Progress(std::size_t _total) :
                     progressC(1ULL), total(_total),
                     fchunk(static_cast<double>(_total) / 100.0),
                     nextChunk(static_cast<std::size_t>(fchunk)) {
    std::cout <<  "     0%"  << std::flush;
}

void Progress::next(std::size_t progress) {
    if (progress == nextChunk) {
        std::cout << "\b\b\b\b\b" << std::setw(4) << progressC++
               << "%" << std::flush;
        nextChunk = static_cast<std::size_t>(
                                   static_cast<double>(progressC) * fchunk);
    }
}

void Progress::perCent(std::size_t progress) {
    std::cout << "\b\b\b\b\b" << std::left << std::setw(4)
           << std::round(xlib::perCent(progress, total))
           << "%" << std::flush;
}

// ------------------------------- Batch ---------------------------------------

Batch::Batch(std::size_t total) : progress(total), partial(0ULL)  {}

} //@xlib
