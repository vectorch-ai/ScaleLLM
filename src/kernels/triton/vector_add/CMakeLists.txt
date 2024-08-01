include(cuda_library)

# find all *.h/*.c files in folder aot recursively
file(GLOB_RECURSE AOT_HEADERS aot/*.h)
file(GLOB_RECURSE AOT_SOURCES aot/*.c)

cuda_library(
  NAME 
    vector_add.kernel
  HDRS 
    ${AOT_HEADERS}
  SRCS 
    ${AOT_SOURCES}
)

