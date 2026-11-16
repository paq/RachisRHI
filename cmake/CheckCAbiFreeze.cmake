cmake_minimum_required(VERSION 3.28)

# A release fixes rachis.h for its RACHIS_ABI_VERSION: while the version stays the same,
# the header may change only in whitespace. This compares HEADER with a baseline copy.
#
#   cmake [-DBASELINE=<git rev>] [-DREQUIRE_BASELINE=ON] -P cmake/CheckCAbiFreeze.cmake
#       Reads the baseline from git. BASELINE defaults to the newest v* tag reachable from
#       HEAD; with no such tag the check passes unless REQUIRE_BASELINE is set.
#   cmake -DHEADER=<file> -DBASELINE_HEADER=<file> [-DBASELINE=<label>] -P ...
#       Compares two files, for the fixture tests.

get_filename_component(source_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT DEFINED HEADER)
    set(HEADER "${source_dir}/include/Rachis/rachis.h")
endif()
if(NOT EXISTS "${HEADER}")
    message(FATAL_ERROR "CheckCAbiFreeze: HEADER '${HEADER}' does not exist.")
endif()
file(READ "${HEADER}" header_text)

if(DEFINED BASELINE_HEADER)
    if(NOT EXISTS "${BASELINE_HEADER}")
        message(FATAL_ERROR "CheckCAbiFreeze: BASELINE_HEADER '${BASELINE_HEADER}' does not exist.")
    endif()
    file(READ "${BASELINE_HEADER}" baseline_text)
    if(NOT DEFINED BASELINE)
        get_filename_component(BASELINE "${BASELINE_HEADER}" NAME)
    endif()
else()
    find_program(git_executable git)
    if(NOT git_executable)
        message(FATAL_ERROR "CheckCAbiFreeze needs git, or -DBASELINE_HEADER=<file>.")
    endif()
    if(NOT DEFINED BASELINE)
        execute_process(
            COMMAND "${git_executable}" describe --tags --abbrev=0 --match "v[0-9]*" HEAD
            WORKING_DIRECTORY "${source_dir}"
            RESULT_VARIABLE describe_status
            OUTPUT_VARIABLE BASELINE
            ERROR_VARIABLE describe_errors
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT describe_status EQUAL 0)
            if(REQUIRE_BASELINE)
                message(FATAL_ERROR
                    "No v* release tag is reachable from HEAD (${describe_errors}). "
                    "A shallow clone has no tags; fetch the full history and tags.")
            endif()
            message(STATUS "No v* release tag is reachable from HEAD, so rachis.h has no baseline.")
            return()
        endif()
    endif()
    file(RELATIVE_PATH header_path "${source_dir}" "${HEADER}")
    execute_process(
        COMMAND "${git_executable}" show "${BASELINE}:${header_path}"
        WORKING_DIRECTORY "${source_dir}"
        RESULT_VARIABLE show_status
        OUTPUT_VARIABLE baseline_text
        ERROR_VARIABLE show_errors)
    if(NOT show_status EQUAL 0)
        message(FATAL_ERROR "Could not read ${header_path} at ${BASELINE}: ${show_errors}")
    endif()
endif()

function(rachis_read_abi_version text label out)
    string(REPLACE "\r\n" "\n" text "${text}")
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" " " text "${text}")
    string(REGEX REPLACE "//[^\n]*" "" text "${text}")
    string(REGEX MATCHALL "#[ \t]*define[ \t]+RACHIS_ABI_VERSION[ \t]+[0-9]+" matches "${text}")
    list(LENGTH matches match_count)
    if(NOT match_count EQUAL 1)
        message(FATAL_ERROR "Expected one #define RACHIS_ABI_VERSION <N> in ${label} (found ${match_count}).")
    endif()
    string(REGEX MATCH "[0-9]+$" version "${matches}")
    set(${out} "${version}" PARENT_SCOPE)
endfunction()

# Splits a header into statements with comments removed and whitespace dropped, except
# between two identifier or number characters. Preprocessor lines keep their line end.
function(rachis_c_abi_statements text out)
    string(REPLACE "\r\n" "\n" text "${text}")
    string(REPLACE ";" "<@SEMI@>" text "${text}")
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" " " text "${text}")
    string(REGEX REPLACE "//[^\n]*" "" text "${text}")
    string(REGEX REPLACE "\\\\\n" " " text "${text}")
    string(REGEX REPLACE "(^|\n)([ \t]*#[^\n]*)" "\\1\\2<@EOL@>" text "${text}")
    string(REGEX REPLACE "[ \t\r\n]+" " " text "${text}")
    string(REGEX REPLACE " ([^A-Za-z0-9_])" "\\1" text "${text}")
    string(REGEX REPLACE "([^A-Za-z0-9_]) " "\\1" text "${text}")
    string(REGEX REPLACE "(<@SEMI@>|<@EOL@>|[{}])" "\\1\n" text "${text}")
    string(REGEX MATCHALL "[^\n]+" statements "${text}")
    set(${out} "${statements}" PARENT_SCOPE)
endfunction()

function(rachis_show_statement statement out)
    string(REPLACE "<@SEMI@>" ";" statement "${statement}")
    string(REPLACE "<@EOL@>" "" statement "${statement}")
    set(${out} "${statement}" PARENT_SCOPE)
endfunction()

rachis_read_abi_version("${header_text}" "${HEADER}" header_version)
rachis_read_abi_version("${baseline_text}" "${BASELINE}" baseline_version)

if(header_version LESS baseline_version)
    message(FATAL_ERROR
        "RACHIS_ABI_VERSION went down from ${baseline_version} in ${BASELINE} to ${header_version}.")
endif()
if(header_version GREATER baseline_version)
    message(STATUS
        "RACHIS_ABI_VERSION is ${header_version}, newer than ${baseline_version} in ${BASELINE}, "
        "so rachis.h may change until version ${header_version} is released.")
    return()
endif()

rachis_c_abi_statements("${header_text}" header_statements)
rachis_c_abi_statements("${baseline_text}" baseline_statements)
if(header_statements STREQUAL baseline_statements)
    message(STATUS "rachis.h matches ${BASELINE} (RACHIS_ABI_VERSION ${header_version}).")
    return()
endif()

list(LENGTH header_statements header_count)
list(LENGTH baseline_statements baseline_count)
set(index 0)
while(index LESS header_count AND index LESS baseline_count)
    list(GET header_statements ${index} header_statement)
    list(GET baseline_statements ${index} baseline_statement)
    if(NOT header_statement STREQUAL baseline_statement)
        break()
    endif()
    math(EXPR index "${index} + 1")
endwhile()
if(index LESS header_count)
    list(GET header_statements ${index} header_statement)
    rachis_show_statement("${header_statement}" header_statement)
else()
    set(header_statement "(end of header)")
endif()
if(index LESS baseline_count)
    list(GET baseline_statements ${index} baseline_statement)
    rachis_show_statement("${baseline_statement}" baseline_statement)
else()
    set(baseline_statement "(end of header)")
endif()

message(FATAL_ERROR
    "rachis.h changed since ${BASELINE} but RACHIS_ABI_VERSION is still ${header_version}.\n"
    "  Header: ${HEADER}\n"
    "  First difference, ignoring whitespace:\n"
    "    ${BASELINE}: ${baseline_statement}\n"
    "    now: ${header_statement}\n"
    "Modules built against ${BASELINE} would misread the changed declarations. Undo the change, or "
    "raise RACHIS_ABI_VERSION in rachis.h and src/C/rachis_abi_exports.txt.")
