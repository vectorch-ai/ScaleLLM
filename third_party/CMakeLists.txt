include(cc_library)

cc_library(
  NAME 
    cutlass
  INCLUDES
    cutlass/include
)

add_subdirectory(sentencepiece)
add_subdirectory(jinja2cpp)

