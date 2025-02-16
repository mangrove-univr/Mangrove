#include "XLib.hpp"
using namespace timer_cuda;

const int SIZE = (1 << 27);
const int BLOCKDIM = 256;

int main() {
    int* devInput;
    cudaMalloc(&devInput, SIZE * sizeof(int));
    Timer<DEVICE> TM;
    TM.start();

    cuda_util::fill<<<SIZE / BLOCKDIM, BLOCKDIM>>>(devInput, 1024, 131072, 1, 1024);

    TM.getTime("fill1");
    __CUDA_ERROR("A")
    TM.start();

    cuda_util::fill2<<<SIZE / BLOCKDIM, BLOCKDIM>>>(devInput, 1024, 131072, 1, 1024);

    TM.getTime("fill2");
    __CUDA_ERROR("B")
}
