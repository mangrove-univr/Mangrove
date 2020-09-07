message("\n==> Test on cmake v" ${CMAKE_VERSION}
        "  System: " ${CMAKE_SYSTEM} "\n")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    add_compile_options("-std=c++11")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

#if (CMAKE_COMPILER_IS_GNUCXX)
    #add_compile_options("-fno-diagnostics-show-caret -ftrack-macro-expansion=0")
#    SET(DEVICE_WARNING_EXTRA
#                        "-fno-diagnostics-show-caret,-ftrack-macro-expansion=0")
#elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#    SET(HOST_WARNING
#    "${HOST_WARNING} -Weverything")
#endif()

if (WARNING_FLAGS)
    #add_compile_options("-Werror")
    add_compile_options("-Wdouble-promotion")
    add_compile_options("-Wformat=2")
    add_compile_options("-Winit-self")
    add_compile_options("-Wmissing-include-dirs")
    add_compile_options("-Wreturn-local-addr")
    #add_compile_options("-Wdate-time")         /gcc v4.9
    add_compile_options("-Wswitch-default")
    add_compile_options("-Wswitch-enum")
    add_compile_options("-Wunused-but-set-variable")
    add_compile_options("-Wunused-local-typedefs")
    add_compile_options("-Wunused")
    #add_compile_options("-Wstrict-overflow=5")
    add_compile_options("-Wundef")
    add_compile_options("-Wendif-labels")
    add_compile_options("-Wshadow")
    add_compile_options("-Wfree-nonheap-object")
    add_compile_options("-Wunsafe-loop-optimizations")
    add_compile_options("-Wpointer-arith")
    add_compile_options("-Wcast-qual")
    add_compile_options("-Wwrite-strings")
    add_compile_options("-Wconversion")
    add_compile_options("-Wsign-conversion")
    add_compile_options("-Wlogical-op")
    add_compile_options("-Wno-aggressive-loop-optimizations")
    add_compile_options("-Wattributes")
    add_compile_options("-Wbuiltin-macro-redefined")
    add_compile_options("-Wmissing-declarations")
    add_compile_options("-Wdeprecated")
    add_compile_options("-Wdeprecated-declarations")
    add_compile_options("-Woverflow")
    add_compile_options("-Wpacked")
    add_compile_options("-Wno-packed-bitfield-compat")
    add_compile_options("-Winline")
    add_compile_options("-Wint-to-pointer-cast")
    add_compile_options("-Winvalid-pch")
    add_compile_options("-Wno-long-long")
    add_compile_options("-Wno-variadic-macros")
    add_compile_options("-Wvarargs")
    add_compile_options("-Wvector-operation-performance")
    add_compile_options("-Wvla")
    add_compile_options("-Wdisabled-optimization")
    add_compile_options("-Wstack-protector")
    add_compile_options("-Woverlength-strings")
    add_compile_options("-Wc++11-compat")
    add_compile_options("-Wconversion-null")
    add_compile_options("-Wuseless-cast")
    add_compile_options("-Winherited-variadic-ctor")
    add_compile_options("-Winvalid-offsetof")
    add_compile_options("-Wvirtual-move-assign")
    add_compile_options("-Wctor-dtor-privacy")
    add_compile_options("-Wdelete-non-virtual-dtor")
    add_compile_options("-Wliteral-suffix")
    add_compile_options("-Wnarrowing")
    add_compile_options("-Wnon-virtual-dtor")
    add_compile_options("-Weffc++")
    add_compile_options("-Wnon-template-friend")
    #add_compile_options("-Wold-style-cast")
    add_compile_options("-Wpmf-conversions")
    add_compile_options("-Wsign-promo")
    add_compile_options("-Wchar-subscripts")
    add_compile_options("-Wno-ignored-qualifiers")
    add_compile_options("-Wdiv-by-zero")
    add_compile_options("-Wtrampolines")
    add_compile_options("-Wfloat-equal")
    add_compile_options("-Wcast-align")
    add_compile_options("-Wsizeof-pointer-memaccess")
    add_compile_options("-Wmultichar")
    add_compile_options("-Wnormalized=nfc")
    add_compile_options("-Wnoexcept")
    add_compile_options("-Wstrict-null-sentinel")
    add_compile_options("-Woverloaded-virtual")
endif()

add_compile_options("-Wall")
add_compile_options("-Wextra")
add_compile_options("-pedantic")

#-------------------------------------------------------------------------------

if (CMAKE_BUILD_TYPE STREQUAL "")
	set (CMAKE_BUILD_TYPE Release)
endif()

message("\n==> Selected CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} "\n")
if (    (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
	AND (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    AND (NOT CMAKE_BUILD_TYPE STREQUAL "Info"))
	message(FATAL_ERROR
    "\nwrong BUILD_TYPE: \"-D CMAKE_BUILD_TYPE=[ Release | Debug | Info ]\"")
endif()

set(XLIB_HOST_SRC
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Host/fUtil.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Host/file_util.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Host/bitmask.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Host/Timer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Host/print_ext.cpp
)
set(XLIB_DEVICE_SRC
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Device/Util/Timer.cu
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Device/Util/cuda_util.cu
    ${CMAKE_CURRENT_LIST_DIR}/../src/Base/Device/Util/global_sync.cu
)
set(GRAPH_HOST_SRC
    ${CMAKE_CURRENT_LIST_DIR}/../src/Graph/Host/GraphBase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/Graph/Host/GraphDegree.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/Graph/Host/GraphSTD.cpp
)
set(GRAPH_DEVICE_SRC
    ${CMAKE_CURRENT_LIST_DIR}/../src/Graph/Device/cudaGraph.cu
    ${CMAKE_CURRENT_LIST_DIR}/../src/Graph/Device/cudaGraphBFS.cu
)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../include/
                    ${CMAKE_CURRENT_LIST_DIR}/../cub-1.4.1/)

string(FIND "${CMAKE_CXX_FLAGS}" "-m32" X86)
#===============================================================================
#===============================================================================
if (CUDA_FOUND)

if (CUDA_VERSION_MAJOR LESS 7)
	message(FATAL_ERROR "\nRequired Cuda Toolkit >= v7.0   Founded: Cuda Toolkit
                        v${CUDA_VERSION_MAJOR}")
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if (WARNING_FLAGS)

#-Wdate-time,\gcc 4.9
#-Wold-style-cast,\

#-Wconversion-null,\
#-Wuseless-cast,\
#-Winherited-variadic-ctor,\
#-Winvalid-offsetof,\
#-Wdelete-non-virtual-dtor,\
#-Wliteral-suffix,\
#-Weffc++,\
#-Wnon-template-friend,\
#-Wsign-promo,\
#-Wnoexcept,\
#-Wpmf-conversions,\
#-Wvirtual-move-assign,\
#-Wstrict-null-sentinel,\
#-Woverloaded-virtual
#-Wc++11-compat,\
#-Wctor-dtor-privacy,\
#-Wnon-virtual-dtor,\

###no cuda
#-Wundef,\
#-Wmissing-declarations,\

### cub
#-Wshadow,\
#-Wconversion,\
set(DEVICE_WARNING_EXTRA
",-Wdouble-promotion,\
-Wformat=2,\
-Winit-self,\
-Wmissing-include-dirs,\
-Wreturn-local-addr,\
-Wswitch-default,\
-Wswitch-enum,\
-Wunused-but-set-variable,\
-Wunused-local-typedefs,\
-Wstrict-overflow=5,\
-Wendif-labels,\
-Wfree-nonheap-object,\
-Wunsafe-loop-optimizations,\
-Wpointer-arith,\
-Wcast-qual,\
-Wwrite-strings,\
-Wsign-conversion,\
-Wlogical-op,\
-Wno-aggressive-loop-optimizations,\
-Wattributes,\
-Wbuiltin-macro-redefined,\
-Wdeprecated,\
-Wdeprecated-declarations,\
-Woverflow,\
-Wpacked,\
-Wno-packed-bitfield-compat,\
-Winline,\
-Wint-to-pointer-cast,\
-Winvalid-pch,\
-Wno-long-long,\
-Wno-variadic-macros,\
-Wvarargs,\
-Wvector-operation-performance,\
-Wvla,\
-Wdisabled-optimization,\
-Wstack-protector,\
-Woverlength-strings,\
-Wnarrowing,\
-Wchar-subscripts,\
-Wno-ignored-qualifiers,\
-Wdiv-by-zero,\
-Wtrampolines,\
-Wfloat-equal,\
-Wcast-align,\
-Wsizeof-pointer-memaccess,\
-Wmultichar,\
-Wnormalized=nfc")
endif()

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11"
    "-Xcompiler=-Wall,-Wextra${DEVICE_WARNING_EXTRA}")

find_library(
	DEVRT_LIBRARY cudadevrt
    HINTS  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

#-------------------------------------------------------------------------------

set(ARCH "" CACHE STRING "compute capability")
set(SM "" CACHE STRING "streaming multiprocessor number")

if(ARCH STREQUAL "")
    if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/../build/getDeviceInfo)
        execute_process(COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc
                    ${CMAKE_CURRENT_LIST_DIR}/getDeviceInfo.cu
                    -o ${CMAKE_CURRENT_LIST_DIR}/../build/getDeviceInfo)
    endif()
    execute_process(COMMAND ${CMAKE_CURRENT_LIST_DIR}/../build/getDeviceInfo
                    OUTPUT_VARIABLE OUT_VARIABLE)

    list(GET OUT_VARIABLE 0 ARCH)
    list(GET OUT_VARIABLE 1 SM)
endif()

if(NOT SM STREQUAL "")
    message("\n==> GPU architecture: ${ARCH}"
            " with ${SM} streaming multiprocessors\n")
else()
    message("\n==> GPU architecture: ${ARCH}\n")
endif()

if (NOT SM STREQUAL "")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DSM=${SM}")
endif()

if (ARCH STREQUAL "30")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30" "-DARCH=300")
elseif (ARCH STREQUAL "32")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_32" "-DARCH=320")
elseif (ARCH STREQUAL "35")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35" "-DARCH=350")
elseif (ARCH STREQUAL "37")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_37" "-DARCH=370")
elseif (ARCH STREQUAL "50")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50" "-DARCH=500")
elseif (ARCH STREQUAL "52")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52" "-DARCH=520")
elseif (ARCH STREQUAL "53")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_53" "-DARCH=530")
else()
	message(FATAL_ERROR "\nplease select GPU architecture:
                        \"cmake -DARCH=[ 30 | 32 | 35 | 37 | 50 | 52 | 53 ]\"")
endif()

# ------------------------------------------------------------------------------

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--use_fast_math"
    "-Xcompiler=-O3,-march=native,-funroll-loops,-DNDEBUG")

elseif (CMAKE_BUILD_TYPE STREQUAL "Info")
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/build/TMP)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-lineinfo" "-keep" "--use_fast_math"
                                "-keep-dir" "${PROJECT_SOURCE_DIR}/build/TMP")

else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g" "-G")
endif()

if (NOT ${X86})
    set(CUDA_64_BIT_DEVICE_CODE OFF)
endif()

endif()
#===============================================================================
#===============================================================================
#debug options

if (NOT ${X86})
    message("==> SET x86\n")
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -ggdb3" )
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif()

#release options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native  -funroll-loops -DNDEBUG")

set(CMAKE_CXX_FLAGS_INFO "-O3 -march=native -funroll-loops")

add_custom_target(rm rm -rf * COMMENT "Clear build directory")

add_custom_target(update rm -rf *
                  COMMAND cmake ..
                  COMMAND make -j 6
                  COMMENT "UPDATE")

add_custom_target(update_debug rm -rf *
                COMMAND cmake -DCMAKE_BUILD_TYPE=Debug ..
                COMMAND make -j 6
                COMMENT "UPDATE DEBUG")

add_custom_target(update_info rm -rf *
                  COMMAND cmake -DCMAKE_BUILD_TYPE=Info ..
                  COMMAND make -j 6
                  COMMENT "UPDATE DEBUG")

#-------------------------------------------------------------------------------
#GRAPHITE="-floop-interchange -ftree-loop-distribution -floop-strip-mine
#            -floop-block"
#CFLAGS="-ftree-vectorize -pipe -flto ${GRAPHITE}"
#LDFLAGS="-Wl,-O1 -Wl,--as-needed,-flto"
