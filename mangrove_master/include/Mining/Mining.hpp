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

#include "DataTypes/TraceProp.cuh"
#include "TemplateConf.cuh"

namespace mangrove {

#if defined(__NVCC__)
extern __device__  entry_t  devDictionary [DictionarySize<numeric_t, MAX_VARS, MAX_ARITY>::value * MAX_ARITY];
extern __device__ result_t  devResult     [DictionarySize<numeric_t, MAX_VARS, MAX_ARITY>::value];
#endif

//==============================================================================

enum TEMPL_ENUM { BT, NB, NT, UNDEFINED };
template<TEMPL_ENUM _enum> struct IndexToTemplate;
template<> struct IndexToTemplate<BT> { using type = BTtemplate; };
template<> struct IndexToTemplate<NB> { using type = NBtemplate; };
template<> struct IndexToTemplate<NT> { using type = NTtemplate; };

template<int _BlockSize,
         int _Steps = 1,
         TEMPL_ENUM _TEMPL_INDEX = UNDEFINED>
struct PROPERTY {
    static_assert(xlib::IS_POWER2<_Steps>::value,
                  PRINT_ERR("PROP::Steps must be a power of Two"));
    static_assert(xlib::IS_POWER2<_BlockSize>::value,
                  PRINT_ERR("PROP::BlockSize must be a power of Two"));

    static const TEMPL_ENUM TEMPL_INDEX = _TEMPL_INDEX;
    static const int BlockSize = _BlockSize;
    static const int Steps = _Steps;
};

} //@mangrove
