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

//#include "include/DataMovementWarp.cuh"
#define ENABLE_CUB

#if defined(ENABLE_CUB)
//#include "DataMovement/DataMovementDynamic.cuh"
//#include "DataMovement/DataMovementQueue.cuh"
#include "DataMovement/DataMovementThread.cuh"
#include "DataMovement/WarpDynamic.cuh"
#endif
#include "DataMovement/RegInsert.cuh"

#include "Primitives/BinarySearches.cuh"
#include "Primitives/cuda_functional.cuh"
#include "Primitives/ThreadReduce.cuh"
#include "Primitives/ThreadScan.cuh"
#include "Primitives/WarpReduce.cuh"
#include "Primitives/WarpScan.cuh"
#include "Primitives/BlockScan.cuh"

#include "Util/cache_load_modifier.cuh"
#include "Util/cache_store_modifier.cuh"
#include "Util/cuda_util.cuh"
#include "Util/definition.cuh"
#include "Util/global_sync.cuh"
#include "Util/basic.cuh"
#include "Util/PTX.cuh"
#include "Util/Timer.cuh"
#include "Util/vector_util.cuh"
