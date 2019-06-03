# ------------------------------------------------------------------------------
# Regression: verify the implementation of Mangrove
# ------------------------------------------------------------------------------

# Helper functions
# ------------------------------------------------------------------------------

function (perform_benchmark_test benchmark_name mining_type conv_option)

  set(outputdir "${PROJECT_SOURCE_DIR}/build/Testing")
  set(basedir   "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark_name}")
  set(orig_trace "${basedir}/${benchmark_name}_${mining_type}.vcd")
  set(conv_trace "${outputdir}/${benchmark_name}_${mining_type}.vcd.mangrove")
  set(variables  "${outputdir}/${benchmark_name}_${mining_type}.vcd.variables")

  add_test(NAME ${benchmark_name}_${mining_type}_conversion
    COMMAND vcd2mangrove ${orig_trace} 1 ${conv_option})

  add_test(NAME ${benchmark_name}_${mining_type}_trace COMMAND mv
    ${orig_trace}.mangrove ${orig_trace}.variables ${outputdir})

  add_test(NAME ${benchmark_name}_${mining_type}_S COMMAND Mangrove -T ${conv_trace}
    -mining=${mining_type} -S -varfile ${variables} -output ${outputdir}/t1.txt)

  add_test(NAME ${benchmark_name}_${mining_type}_M COMMAND Mangrove -T ${conv_trace}
    -mining=${mining_type} -M -varfile ${variables} -output ${outputdir}/t2.txt)

  if(NOT ${ONLY_HOST})

    #message("----------------------------------------------------problem")

    add_test(NAME ${benchmark_name}_${mining_type}_GPU COMMAND Mangrove -T ${conv_trace}
     -mining=${mining_type} -GPU -varfile ${variables} -output ${outputdir}/t3.txt)

  endif()

  add_test(NAME ${benchmark_name}_${mining_type}_diff_SM COMMAND
    diff ${outputdir}/t1.txt ${outputdir}/t2.txt)

  if(NOT ${ONLY_HOST})

    add_test(NAME ${benchmark_name}_${mining_type}_diff_SGPU COMMAND
      diff ${outputdir}/t1.txt ${outputdir}/t3.txt)

  endif()

  add_test(NAME ${benchmark_name}_${mining_type}_inf_S COMMAND Mangrove_inference_all
    -T ${conv_trace} -mining=${mining_type} -S -varfile ${variables}
    -output ${outputdir}/inf_t1.txt)

  add_test(NAME ${benchmark_name}_${mining_type}_inf_M COMMAND Mangrove_inference_all
    -T ${conv_trace} -mining=${mining_type} -M -varfile ${variables}
    -output ${outputdir}/inf_t2.txt)

  if(NOT ${ONLY_HOST})

    add_test(NAME ${benchmark_name}_${mining_type}_inf_GPU COMMAND Mangrove_inference_all
      -T ${conv_trace} -mining=${mining_type} -GPU -varfile ${variables}
      -output ${outputdir}/inf_t3.txt)

  endif()

  add_test(NAME ${benchmark_name}_${mining_type}_inf_diff_SM COMMAND
    diff ${outputdir}/inf_t1.txt ${outputdir}/inf_t2.txt)

  if(NOT ${ONLY_HOST})

    add_test(NAME ${benchmark_name}_${mining_type}_inf_diff_SGPU COMMAND
      diff ${outputdir}/inf_t1.txt ${outputdir}/inf_t3.txt)

  endif()

  add_test(NAME ${benchmark_name}_${mining_type}_inf_diff COMMAND
    diff ${outputdir}/t1.txt ${outputdir}/inf_t1.txt)

  add_test(NAME ${benchmark_name}_${mining_type}_MMAPparsing COMMAND Mangrove -T ${conv_trace}
    -mining=${mining_type} -read=MMAP -check-read -varfile ${variables})

  if(NOT ${ONLY_HOST})

    add_test(NAME ${benchmark_name}_${mining_type}_GPUparsing COMMAND Mangrove -T ${conv_trace}
      -mining=${mining_type} -read=GPU -check-read -varfile ${variables})

  endif()

endfunction (perform_benchmark_test)

function (add_benchmark_test benchmark_name bool_test num_test)

  if(${bool_test})

    message("-- Configuring boolean test for ${benchmark_name}")
    perform_benchmark_test(${benchmark_name} bool 1)

  endif()

  if(${num_test})

    message("-- Configuring numeric test for ${benchmark_name}")
    perform_benchmark_test(${benchmark_name} numeric 2)

  endif()

endfunction (add_benchmark_test)

# Add benchmarks
# ------------------------------------------------------------------------------

add_benchmark_test(apb YES YES)
add_benchmark_test(fpadd YES YES)
add_benchmark_test(fpmul YES YES)
add_benchmark_test(uart YES YES)
add_benchmark_test(bool YES NO)
add_benchmark_test(numeric NO YES)
add_benchmark_test(jpeg_enc YES YES)
add_benchmark_test(jpeg_dct YES YES)
add_benchmark_test(jpeg_qnr YES YES)
add_benchmark_test(jpeg_rle YES YES)
add_benchmark_test(jpeg_zig YES YES)
