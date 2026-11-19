# Runs slangc to write a reflection JSON and writes a Makefile depfile that lists every source file
# slangc read, includes and imported modules alike, so the reflection reruns when any of them changes.
#
#   cmake -DTARGET=<reflection.json> -DDEPFILE=<file.d> -P WriteShaderReflection.cmake -- <slangc> <argument>...
#
# slangc writes no depfile without code generation, so the script adds -output-includes and reads the
# files from the notes slangc prints; any other diagnostic is passed through.
cmake_minimum_required(VERSION 3.28)

foreach(required_variable IN ITEMS TARGET DEPFILE)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(command)
set(separator_found FALSE)
math(EXPR last_argument "${CMAKE_ARGC} - 1")
foreach(index RANGE ${last_argument})
    if(separator_found)
        list(APPEND command "${CMAKE_ARGV${index}}")
    elseif("${CMAKE_ARGV${index}}" STREQUAL "--")
        set(separator_found TRUE)
    endif()
endforeach()
if(NOT command)
    message(FATAL_ERROR "Pass the slangc command after --.")
endif()

execute_process(
    COMMAND ${command} -output-includes
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE diagnostics)

string(REPLACE "\r\n" "\n" diagnostics "${diagnostics}")
string(REPLACE ";" "\\;" diagnostics "${diagnostics}")
string(REPLACE "\n" ";" lines "${diagnostics}")
set(dependencies)
set(passed_through "")
foreach(line IN LISTS lines)
    if(line MATCHES "note: include +'(.+)'$")
        list(APPEND dependencies "${CMAKE_MATCH_1}")
    elseif(NOT line STREQUAL "")
        string(APPEND passed_through "${line}\n")
    endif()
endforeach()
if(NOT output STREQUAL "")
    message(NOTICE "${output}")
endif()
if(NOT passed_through STREQUAL "")
    message(NOTICE "${passed_through}")
endif()
if(NOT status EQUAL 0)
    message(FATAL_ERROR "slangc failed to reflect the shader (${status}).")
endif()
if(NOT dependencies)
    message(FATAL_ERROR "slangc -output-includes listed no source files, so the reflection has no depfile.")
endif()

function(rachis_depfile_path output path)
    if(CMAKE_HOST_WIN32)
        string(REPLACE "\\" "/" path "${path}")
    endif()
    cmake_path(ABSOLUTE_PATH path NORMALIZE)
    string(REPLACE "$" "$$" path "${path}")
    string(REPLACE "#" "\\#" path "${path}")
    string(REPLACE " " "\\ " path "${path}")
    set(${output} "${path}" PARENT_SCOPE)
endfunction()

rachis_depfile_path(content "${TARGET}")
string(APPEND content ":")
set(written)
foreach(dependency IN LISTS dependencies)
    rachis_depfile_path(dependency "${dependency}")
    if(NOT dependency IN_LIST written)
        list(APPEND written "${dependency}")
        string(APPEND content " \\\n  ${dependency}")
    endif()
endforeach()
file(WRITE "${DEPFILE}" "${content}\n")
