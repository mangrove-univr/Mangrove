nvcc -I../../ -I../../../cub-1.4.1 -arch=sm_35 -std=c++11 -keep -keep-dir=./TMP  -use_fast_math ../../Base/Host/src/Timer.cpp ../../Base/Device/Util/src/Timer.cu OptimizedFill.cu  -o OptimizedFill
