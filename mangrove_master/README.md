
# How to install #

* git clone .....
* cd build
* cmake ..
* make -j 6

Tested on Ubuntu 14.04/16.04, CUDA 7.0/10.0, gcc-4.8 / gcc-5.5, GeForce 780 GTX

# How to use #

* Mangrove usage : see "Syntax.txt"

# Example #

* cd build
* vcd2mangrove/vcd2mangrove ../regression/jpeg_qnr/jpeg_qnr_numeric.vcd 1 2
* ./Mangrove -T ../regression/jpeg_qnr/jpeg_qnr_numeric.vcd.mangrove -mining=numeric -S -varfile ../regression/jpeg_qnr/jpeg_qnr_numeric.vcd.variables -output t1.txt

### Regression test ###

* make test

### Other commands ###

* make rm : remove all files
* make update : remove all files, execute cmake, execute make
* make update_debug : debug mode
* make update_info : info mode

# To do and future work  #

* LOP3 optimization for 4-Arity boolean functions on Maxwell architectures
* GPU parsing of float trace
* Time window Invariant mining
* Entropy Analysis
* Pattern Recognition / String Matching
* FSM Deduction on small windows
* Graph Inference on Numeric Variables

* Numeric invariants tested with this:
./Mangrove_inference_all -mining=numeric -G 14 1000 -range 10_10,10_10,12_13,13_14,15_16,16_17,18_20,19_21,22_24,23_25,26_27,28_29,30_31,32_33 -S
