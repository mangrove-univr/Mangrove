MANGROVE_DIR="/storage/lavanzini/TEST/mangrove/build"
TRACE_DIR="/storage/lavanzini/SSE/testbenchs/Test"
RESULT_DIR="/storage/lavanzini/TEST/Results/Test"

MANGROVE_NO_INFERENCE="$MANGROVE_DIR/Mangrove"
MANGROVE_INFERENCE="$MANGROVE_DIR/Mangrove_inference"
MANGROVE_INFERENCE_ALL="$MANGROVE_DIR/Mangrove_inference_all"

rm $RESULT_DIR/*.txt

#NO INFERENCE
echo "Mangrove no inference - Host"
$MANGROVE_NO_INFERENCE -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -S -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/NoInference_Host.txt 1>&2 > $RESULT_DIR/Mangrove_no_inference.txt

echo "Mangrove no inference - Thread"
$MANGROVE_NO_INFERENCE -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -M -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/NoInference_Threads.txt 1>&2 >> $RESULT_DIR/Mangrove_no_inference.txt

echo "Mangrove no inference - GPU"
$MANGROVE_NO_INFERENCE -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -GPU -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/NoInference_GPU.txt 1>&2 >> $RESULT_DIR/Mangrove_no_inference.txt


#INFERENCE
# echo "Mangrove inference - Host"
# $MANGROVE_INFERENCE -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -S -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/Inference_Host.txt 1>&2 > $RESULT_DIR/Mangrove_inference.txt

# echo "Mangrove inference - Thread"
# $MANGROVE_INFERENCE -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -M -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/Inference_Threads.txt 1>&2 >> $RESULT_DIR/Mangrove_inference.txt

# echo "Mangrove inference - GPU"
# $MANGROVE_INFERENCE -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -GPU -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/Inference_GPU.txt 1>&2 >> $RESULT_DIR/Mangrove_inference.txt


#INFERENCE ALL
echo "Mangrove inference all - Host"
$MANGROVE_INFERENCE_ALL -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -S -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/Inference_all_Host.txt 1>&2 > $RESULT_DIR/Mangrove_inference_all.txt

echo "Mangrove inference all - Thread"
$MANGROVE_INFERENCE_ALL -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -M -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/Inference_all_Threads.txt 1>&2 >> $RESULT_DIR/Mangrove_inference_all.txt

echo "Mangrove inference all - GPU"
$MANGROVE_INFERENCE_ALL -T $TRACE_DIR/Test-trace.vcd.mangrove -mining=numeric -GPU -varfile $TRACE_DIR/Test-trace.vcd.variables -output $RESULT_DIR/Inference_all_GPU.txt 1>&2 >> $RESULT_DIR/Mangrove_inference_all.txt
