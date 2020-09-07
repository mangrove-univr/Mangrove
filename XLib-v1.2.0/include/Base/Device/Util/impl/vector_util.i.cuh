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

namespace std {

template<> class numeric_limits<int2> {
     public:
    static int2 max() {
        return make_int2(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
    }
};

 template<> class numeric_limits<int3> {
     public:
    static int3 max() {
        return make_int3(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
    }
};

template<> class numeric_limits<int4> {
    public:
    static int4 max() {
        return make_int4(std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::max());
    }
};

}

inline std::ostream& operator<<(std::ostream& out, const int2& value) {
    out << "( " << value.x << "," << value.y << " )";
    return out;
}

inline bool operator== (const int2& A, const int2& B) {
    return A.x == B.x && A.y == B.y;
}

inline bool operator!= (const int2& A, const int2& B) {
    return A.x != B.x || A.y != B.y;
}

inline bool operator< (const int2& A, const int2& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y);
}

inline bool operator<= (const int2& A, const int2& B) {
    return A.x <= B.x && A.y <= B.y;
}

inline bool operator>= (const int2& A, const int2& B) {
    return A.x >= B.x && A.y >= B.y;
}

inline bool operator> (const int2& A, const int2& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y);
}

//------------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& out, const int4& value) {
    out << "( " << value.x << "," << value.y << ","
        << value.z << "," << value.w << " )";
    return out;
}

inline bool operator== (const int4& A, const int4& B) {
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

inline bool operator!= (const int4& A, const int4& B) {
    return A.x != B.x || A.y != B.y || A.z != B.z || A.w != B.w;
}

inline bool operator< (const int4& A, const int4& B) {
    return A.x < B.x || (A.x == B.x && A.y < B.y) ||
                        (A.y == B.y && A.z < B.z) ||
                        (A.z == B.z && A.w < B.w);
}

inline bool operator<= (const int4& A, const int4& B) {
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}

inline bool operator>= (const int4& A, const int4& B) {
    return A.x >= B.x && A.y >= B.y & A.z >= B.z & A.w >= B.w;
}

inline bool operator> (const int4& A, const int4& B) {
    return A.x > B.x || (A.x == B.x && A.y > B.y)
                     || (A.y == B.y && A.z > B.z)
                     || (A.z == B.z && A.w > B.w);
}
