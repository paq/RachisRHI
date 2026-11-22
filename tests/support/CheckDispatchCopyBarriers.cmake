cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED TEST_ROOT OR NOT IS_DIRECTORY "${TEST_ROOT}")
    message(FATAL_ERROR "CheckDispatchCopyBarriers requires -DTEST_ROOT=<tests directory> (got '${TEST_ROOT}').")
endif()

file(GLOB_RECURSE sources LIST_DIRECTORIES false
    "${TEST_ROOT}/*.c"
    "${TEST_ROOT}/*.cpp"
    "${TEST_ROOT}/*.mm")
list(SORT sources)

set(dispatch_then_copy
    "[.>]dispatch[A-Za-z]*\\([^;]*;[ \t\r\n]*[^;]*[.>]copy(Buffer|TextureToBuffer|BufferToTexture|Texture)[A-Za-z]*\\(")

set(offenders)
foreach(source IN LISTS sources)
    file(READ "${source}" content)
    file(RELATIVE_PATH relative "${TEST_ROOT}" "${source}")
    while(TRUE)
        string(REGEX MATCH "${dispatch_then_copy}" match "${content}")
        if(match STREQUAL "")
            break()
        endif()
        string(FIND "${content}" "${match}" position)
        string(LENGTH "${match}" length)
        math(EXPR position "${position} + ${length}")
        string(SUBSTRING "${content}" ${position} -1 content)
        string(REGEX REPLACE "[ \t\r\n]+" " " match "${match}")
        string(REPLACE ";" "" match "${match}")
        list(APPEND offenders "${relative}: ${match}")
    endwhile()
endforeach()

if(offenders)
    list(JOIN offenders "\n  " report)
    message(FATAL_ERROR
        "These tests copy a resource right after a dispatch with no barrier in between, so the copy can race "
        "the shader write:\n  ${report}\n"
        "Record a bufferBarrier/textureBarrier (computeShader -> copy) before the copy.")
endif()

list(LENGTH sources source_count)
message(STATUS "Checked ${source_count} test sources for copies recorded right after a dispatch.")
