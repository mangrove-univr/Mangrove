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

namespace xlib {

enum class QueuePolicy {FIFO, LIFO};

template<typename T>
class Queue {
private:
    int left, right, N;
    T* Array;
    Queue<T>(const Queue<T>&) = delete;
    void operator=(const Queue&) = delete;

public:
    Queue();
    Queue(int _size);
    ~Queue();

    void init(int _size);
    void free();
    void reset();
    void insert(T value);

    template<QueuePolicy POLICY = QueuePolicy::FIFO>
    T extract();
    T extract(int i);

    bool isEmpty()  const;
    int size()      const;
    int totalSize() const;
    T at(int i)     const;
    T last()        const;
    T get(int i)    const;

    void sort();
    void print()    const;
};

} //@xlib

#include "impl/fast_queue.i.hpp"
