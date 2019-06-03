set(PROJECT "Mangrove")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.2)
FIND_PACKAGE(CUDA)

# ------------------------------------------------------------------------------

set(XLIB_DIR    ${PROJECT_SOURCE_DIR}/../XLib-v1.2)

if (NOT EXISTS "${XLIB_DIR}")
    message(FATAL_ERROR "\nXLib not found : ${XLIB_DIR}")
endif()

#set(WARNING_FLAGS ON)
set(NOT_FLOAT_EQUAL ON)

set(ONLY_HOST OFF)
set(INFERENCE ON)
set(AUTO_TUNING ON)
set(NO_CONSTANT ON)

include(${XLIB_DIR}/CMakeBase/CMakeLists.txt)

# ------------------------------------------------------------------------------
# VCD2MANGROVE
add_subdirectory(vcd2mangrove)

# ------------------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${PROJECT_SOURCE_DIR}/TemplateConf/
                    ${XLIB_DIR}/include/
                    ${XLIB_DIR}/cub-1.4.1/
)

set(INFERENCE_SRC
    ${PROJECT_SOURCE_DIR}/src/Inference/ResultCollectorBool.cpp
    ${PROJECT_SOURCE_DIR}/src/Inference/ResultCollectorNumeric.cpp
    ${PROJECT_SOURCE_DIR}/src/Inference/GeneratorBool.cpp
    ${PROJECT_SOURCE_DIR}/src/Inference/GeneratorNumeric.cpp)

set(PRINTER_SRC
    ${PROJECT_SOURCE_DIR}/src/Printer/InvariantPrinterBool.cpp
    ${PROJECT_SOURCE_DIR}/src/Printer/InvariantPrinterNumeric.cpp)

set(GET_TRACE_SRC
    ${PROJECT_SOURCE_DIR}/src/GetTrace/ReadTrace.cu
    ${PROJECT_SOURCE_DIR}/src/GetTrace/GenerateTrace.cu
    ${PROJECT_SOURCE_DIR}/src/GetTrace/GetTraceSupport.cpp)

set(MINING_HOST_SRC
    ${PROJECT_SOURCE_DIR}/src/DataTypes/ModuleSTR.cpp
    ${PROJECT_SOURCE_DIR}/src/DataTypes/TraceProp.cu
    ${PROJECT_SOURCE_DIR}/src/Mining/Host/HostMiningFixedBool.cu
    ${PROJECT_SOURCE_DIR}/src/Mining/Host/HostMiningFixedNumeric.cu)

set(MAIN_SRC
    ${PROJECT_SOURCE_DIR}/src/Utility.cpp)

set(CUDA_SEPARABLE_COMPILATION ON)

#===============================================================================

if (${NO_CONSTANT})
    set(NO_CONST_FLAG -DNOCONSTANT)
endif()

if (${ONLY_HOST})

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pthread -Wl,--no-as-needed")

add_executable(Mangrove
                ${PROJECT_SOURCE_DIR}/Mangrove.cu ${MAIN_SRC}
                ${INFERENCE_SRC} ${PRINTER_SRC} ${GET_TRACE_SRC}
                ${MINING_HOST_SRC} ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})

set_target_properties(Mangrove PROPERTIES
                      COMPILE_FLAGS "${NO_CONST_FLAG} -x c++")
set_target_properties(Mangrove PROPERTIES LINKER_LANGUAGE CXX)

if(${INFERENCE})

add_executable(Mangrove_inference_all
                ${PROJECT_SOURCE_DIR}/Mangrove.cu ${MAIN_SRC}
                ${INFERENCE_SRC} ${PRINTER_SRC} ${GET_TRACE_SRC}
                ${MINING_HOST_SRC} ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC})

set_target_properties(Mangrove_inference_all PROPERTIES COMPILE_FLAGS
"-DBOOL_INFERENCE -DNUMERIC_INFERENCE -DGET_ALL_INVARIANTS ${NO_CONST_FLAG} -x c++")
set_target_properties(Mangrove_inference_all PROPERTIES LINKER_LANGUAGE CXX)
endif()

set_source_files_properties(${PROJECT_SOURCE_DIR}/Mangrove.cu ${MAIN_SRC}
                           ${INFERENCE_SRC} ${GET_TRACE_SRC} ${MINING_HOST_SRC}
                           ${XLIB_HOST_SRC} PROPERTIES LANGUAGE CXX)

endif()

#===============================================================================

if(CUDA_FOUND AND NOT ${ONLY_HOST})

set(MINING_DEVICE_SRC
            ${PROJECT_SOURCE_DIR}/src/Mining/Device/AutoTuning.cu
            ${PROJECT_SOURCE_DIR}/src/Mining/Device/GPUMiningFixedBool.cu
            ${PROJECT_SOURCE_DIR}/src/Mining/Device/GPUMiningFixedNumeric.cu)

# ------------------------------------------------------------------------------

if(AUTO_TUNING)
    set(AUTO_TUNING_VAR -DAUTO_TUNING)
endif()

cuda_add_executable(Mangrove ${PROJECT_SOURCE_DIR}/Mangrove.cu ${MAIN_SRC}
                    ${INFERENCE_SRC} ${PRINTER_SRC} ${GET_TRACE_SRC}
                    ${MINING_HOST_SRC}
                    ${MINING_DEVICE_SRC} ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC}
                    OPTIONS ${AUTO_TUNING_VAR} ${NO_CONST_FLAG})

set_target_properties(Mangrove PROPERTIES COMPILE_FLAGS "${NO_CONST_FLAG}")

if(${INFERENCE})

cuda_add_executable(Mangrove_inference_all ${PROJECT_SOURCE_DIR}/Mangrove.cu
                  ${MAIN_SRC} ${INFERENCE_SRC} ${PRINTER_SRC}
                  ${GET_TRACE_SRC} ${MINING_HOST_SRC}
                  ${MINING_DEVICE_SRC} ${XLIB_HOST_SRC} ${XLIB_DEVICE_SRC}
                  OPTIONS -DBOOL_INFERENCE -DNUMERIC_INFERENCE ${NO_CONST_FLAG}
                          -DGET_ALL_INVARIANTS ${AUTO_TUNING_VAR})

set_target_properties(Mangrove_inference_all
PROPERTIES COMPILE_FLAGS
   "-DBOOL_INFERENCE -DNUMERIC_INFERENCE -DGET_ALL_INVARIANTS ${NO_CONST_FLAG}")
endif()
endif()

enable_testing()
add_subdirectory(regression)
