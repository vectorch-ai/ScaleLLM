include(cc_library)

cc_library(
  NAME 
    cutlass
  INCLUDES
    cutlass/include
)

cc_library(
  NAME 
    flashinfer
  INCLUDES
    flashinfer/include
)

add_subdirectory(sentencepiece)
add_subdirectory(jinja2cpp)

