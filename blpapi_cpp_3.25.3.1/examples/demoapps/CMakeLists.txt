cmake_minimum_required(VERSION 3.17.0)
project(demoapps LANGUAGES CXX)

# If no CMAKE_PREFIX_PATH was provided, add the expected location
# relative to the examples
if (NOT CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")

if (NOT CMAKE_CXX_STANDARD OR
    CMAKE_CXX_STANDARD LESS 14 OR
    CMAKE_CXX_STANDARD EQUAL 98)
  message(FATAL_ERROR "CMAKE_CXX_STANDARD>=14 is required")
endif ()
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Blpapi
find_package(blpapi REQUIRED CONFIG)

if(NOT WIN32)
  # The blpapi shared object currently does not define an "soname", and we
  # need to inform cmake of that. Otherwise, the linker will use the file
  # path to the shared object as the name in the dependency list, which will
  # later on cause the loader to exclusively use that file path and disregard
  # 'LD_LIBRARY_PATH' (this is the behavior of the loader when it sees a '/'
  # in the name of a dependency).
  set_property(TARGET blpapi PROPERTY IMPORTED_NO_SONAME TRUE)
endif()

# If not specified, build for current machine architecture.
math(EXPR _ARCH "8 * ${CMAKE_SIZEOF_VOID_P}")

# Configure the specified 'example'
function(configure_example_target example)
  add_executable(${example} ${example}.cpp)
  target_include_directories(
    ${example} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  set_target_properties(${example} PROPERTIES OUTPUT_NAME "${example}_${_ARCH}")

  if(WIN32)
    target_compile_definitions(${example} PUBLIC _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_definitions(${example} PUBLIC _POSIX_PTHREAD_SEMANTICS)

  endif()

  target_link_libraries(${example} PUBLIC blpapi)

  install(TARGETS ${example}
    DESTINATION "${OUTPUT_DIR}"
    COMPONENT "demoapps")
endfunction()

file(STRINGS "examples.list" EXAMPLES)

foreach(example ${EXAMPLES})
  configure_example_target(${example})
endforeach(${example})
