# ------------------------------------------------------------------------------
# VCD2MANGROVE
# ------------------------------------------------------------------------------

#includes
include_directories(include)

# sources
set (VCD2MANGROVE_SRC
  ${PROJECT_SOURCE_DIR}/vcd2mangrove/src/vcd2mangrove_main.cc
  ${PROJECT_SOURCE_DIR}/vcd2mangrove/src/Vcd2mangrove.cc)

# executable
add_executable(vcd2mangrove ${VCD2MANGROVE_SRC})
