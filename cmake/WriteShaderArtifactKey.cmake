cmake_minimum_required(VERSION 3.28)

foreach(required_variable IN ITEMS OUTPUT SOURCE COMPILER TARGET PROFILE ENTRY ENVIRONMENT OPTIONS)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

if(NOT EXISTS "${SOURCE}")
    message(FATAL_ERROR "Shader source '${SOURCE}' does not exist")
endif()

file(SHA256 "${SOURCE}" source_hash)
# A file name in the depfile may contain a semicolon, which would split a CMake list element; carry it as the
# ASCII unit separator while the names are in a list.
string(ASCII 31 semicolon)
set(include_paths)
if(DEFINED INCLUDES)
    list(APPEND include_paths ${INCLUDES})
endif()
if(DEFINED DEPFILE AND NOT DEPFILE STREQUAL "")
    if(NOT EXISTS "${DEPFILE}")
        message(FATAL_ERROR "Shader depfile '${DEPFILE}' does not exist")
    endif()
    set(source_path "${SOURCE}")
    if(CMAKE_HOST_WIN32)
        string(REPLACE "\\" "/" source_path "${source_path}")
    endif()
    cmake_path(ABSOLUTE_PATH source_path NORMALIZE)
    file(READ "${DEPFILE}" depfile_text)
    string(REPLACE ";" "${semicolon}" depfile_text "${depfile_text}")
    string(REGEX REPLACE "[ \t]*\\\\\r?\n[ \t]*" "\n" depfile_text "${depfile_text}")
    string(REGEX MATCHALL "[^\n]+" depfile_lines "${depfile_text}")
    list(REMOVE_AT depfile_lines 0)
    foreach(include_path IN LISTS depfile_lines)
        string(STRIP "${include_path}" include_path)
        string(REPLACE "\\ " " " include_path "${include_path}")
        string(REPLACE "\\#" "#" include_path "${include_path}")
        string(REPLACE "$$" "$" include_path "${include_path}")
        string(REPLACE "${semicolon}" ";" decoded_path "${include_path}")
        if(NOT include_path STREQUAL "" AND NOT decoded_path STREQUAL source_path)
            list(APPEND include_paths "${include_path}")
        endif()
    endforeach()
endif()
set(include_lines "")
foreach(include_path IN LISTS include_paths)
    string(REPLACE "${semicolon}" ";" include_path "${include_path}")
    if(NOT EXISTS "${include_path}")
        message(FATAL_ERROR "Shader include '${include_path}' does not exist")
    endif()
    file(SHA256 "${include_path}" include_hash)
    string(APPEND include_lines "include=${include_path}\ninclude_hash=${include_hash}\n")
endforeach()

if(NOT DEFINED DEFINES)
    set(DEFINES "")
endif()

set(extra_lines "")
if(DEFINED COMPILER_VERSION AND NOT COMPILER_VERSION STREQUAL "")
    string(APPEND extra_lines "compiler_version=${COMPILER_VERSION}\n")
endif()
if(DEFINED SDK AND NOT SDK STREQUAL "")
    string(APPEND extra_lines "sdk=${SDK}\n")
endif()
if(DEFINED MSL AND NOT MSL STREQUAL "")
    string(APPEND extra_lines "msl=${MSL}\n")
endif()

file(WRITE "${OUTPUT}"
    "source=${SOURCE}\n"
    "source_hash=${source_hash}\n"
    "${include_lines}"
    "compiler=${COMPILER}\n"
    "target=${TARGET}\n"
    "profile=${PROFILE}\n"
    "entry=${ENTRY}\n"
    "environment=${ENVIRONMENT}\n"
    "options=${OPTIONS}\n"
    "defines=${DEFINES}\n"
    "${extra_lines}"
)
