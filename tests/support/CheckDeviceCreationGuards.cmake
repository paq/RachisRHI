cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED TEST_ROOT OR NOT IS_DIRECTORY "${TEST_ROOT}")
    message(FATAL_ERROR "CheckDeviceCreationGuards requires -DTEST_ROOT=<tests directory> (got '${TEST_ROOT}').")
endif()

file(GLOB_RECURSE sources LIST_DIRECTORIES false
    "${TEST_ROOT}/*.c"
    "${TEST_ROOT}/*.cpp"
    "${TEST_ROOT}/*.h"
    "${TEST_ROOT}/*.mm")
list(SORT sources)

set(adapter_not_found_guard
    "(CHECK|REQUIRE)\\([^;]*!=[ \t\r\n]*(rachis::)?ErrorCode::adapterNotFound[ \t\r\n]*\\)")
set(bare_creation_return
    "if[ \t]*\\([ \t]*![ \t]*[A-Za-z_]*[Cc]reated[ \t]*\\)[ \t\r\n]*[{]?[ \t\r\n]*return[ \t\r\n]*;")

set(offenders)
foreach(source IN LISTS sources)
    file(READ "${source}" content)
    file(RELATIVE_PATH relative "${TEST_ROOT}" "${source}")
    foreach(pattern IN ITEMS adapter_not_found_guard bare_creation_return)
        string(REGEX MATCH "${${pattern}}" match "${content}")
        if(NOT match STREQUAL "")
            string(REGEX REPLACE "[ \t\r\n]+" " " match "${match}")
            string(REPLACE ";" "" match "${match}")
            list(APPEND offenders "${relative}: ${match}")
        endif()
    endforeach()
endforeach()

if(offenders)
    list(JOIN offenders "\n  " report)
    message(FATAL_ERROR
        "These test files guard a failed device creation with only an adapterNotFound check or a bare "
        "return, so the case can pass without running:\n  ${report}\n"
        "Use rachis::test::skipWithoutCreatedDevice from RhiTestSupport.h, which skips or fails instead.")
endif()

list(LENGTH sources source_count)
message(STATUS "Checked ${source_count} test sources for vacuous device-creation guards.")
