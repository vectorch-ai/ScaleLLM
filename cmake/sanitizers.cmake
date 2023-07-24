set(USE_SANITIZER "" CACHE STRING "Enable sanitizer(s). Options are: address, leak, memory, thread, undefined. Case insensitive; multiple options delimited by comma or space possible.")

string(TOLOWER "${USE_SANITIZER}" USE_SANITIZER)

if(USE_SANITIZER MATCHES "address")
  list(APPEND SANITIZER_COMPILE_OPTIONS -fsanitize=address -fno-omit-frame-pointer)
  list(APPEND SANITIZER_LINK_OPTIONS -fsanitize=address -fno-omit-frame-pointer)
endif()

if(USE_SANITIZER MATCHES "leak")
  list(APPEND SANITIZER_COMPILE_OPTIONS -fsanitize=leak -fno-omit-frame-pointer)
  list(APPEND SANITIZER_LINK_OPTIONS -fsanitize=leak -fno-omit-frame-pointer)
endif()

if(USE_SANITIZER MATCHES "memory")
  list(APPEND SANITIZER_COMPILE_OPTIONS -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer -fPIE -pie)
  list(APPEND SANITIZER_LINK_OPTIONS -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer -fPIE -pie)
endif()

if(USE_SANITIZER MATCHES "thread")
  list(APPEND SANITIZER_COMPILE_OPTIONS -fsanitize=thread -fno-omit-frame-pointer)
  list(APPEND SANITIZER_LINK_OPTIONS -fsanitize=thread -fno-omit-frame-pointer)
endif()

if(USE_SANITIZER MATCHES "undefined")
  list(APPEND SANITIZER_COMPILE_OPTIONS -fsanitize=undefined -fno-omit-frame-pointer)
  list(APPEND SANITIZER_LINK_OPTIONS -fsanitize=undefined -fno-omit-frame-pointer)
endif()

list(REMOVE_DUPLICATES SANITIZER_COMPILE_OPTIONS)
list(REMOVE_DUPLICATES SANITIZER_LINK_OPTIONS)

if(SANITIZER_COMPILE_OPTIONS)
  message(STATUS "Using sanitizer: " ${USE_SANITIZER})
  add_compile_options(${SANITIZER_COMPILE_OPTIONS})
  add_link_options(${SANITIZER_LINK_OPTIONS})
endif()