foreach(required IN ITEMS RACHIS_SOURCE_DIR SOURCE_DIR WORK_DIR CONFIG GENERATOR CTEST_COMMAND SYSTEM_NAME
        SYSTEM_PROCESSOR)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required.")
    endif()
endforeach()

set(arguments -G "${GENERATOR}"
    "-DRACHIS_SOURCE_DIR=${RACHIS_SOURCE_DIR}"
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON)
set(settings
    MAKE_PROGRAM:CMAKE_MAKE_PROGRAM
    C_COMPILER:CMAKE_C_COMPILER
    CXX_COMPILER:CMAKE_CXX_COMPILER
    RC_COMPILER:CMAKE_RC_COMPILER
    MT_TOOL:CMAKE_MT
    OSX_ARCHITECTURES:CMAKE_OSX_ARCHITECTURES
    OSX_DEPLOYMENT_TARGET:CMAKE_OSX_DEPLOYMENT_TARGET
    DXC:RACHIS_DXC_EXECUTABLE
    DXC_SPIRV:RACHIS_DXC_SPIRV_EXECUTABLE
    SPIRV_VAL:RACHIS_SPIRV_VAL_EXECUTABLE
    SHADER_BINARY_TOOL:RACHIS_SHADER_BINARY_TOOL
    BACKENDS:RACHIS_BACKENDS
    DEBUG_MARKERS:RACHIS_ENABLE_DEBUG_MARKERS)
set(environment)
if(WITHOUT_SLANGC)
    list(APPEND arguments -DRACHIS_PACKAGE_WITHOUT_SLANGC=ON)
    set(environment "${CMAKE_COMMAND}" -E env --unset=SLANGC)
else()
    list(APPEND settings SLANGC:RACHIS_SLANGC_EXECUTABLE)
endif()
foreach(setting IN LISTS settings)
    string(REPLACE ":" ";" setting "${setting}")
    list(GET setting 0 input)
    list(GET setting 1 variable)
    if(NOT "${${input}}" STREQUAL "")
        list(APPEND arguments "-D${variable}=${${input}}")
    endif()
endforeach()
get_cmake_property(defined_variables VARIABLES)
foreach(variable IN LISTS defined_variables)
    if(variable MATCHES "^FETCHCONTENT_SOURCE_DIR_")
        list(APPEND arguments "-D${variable}=${${variable}}")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(build "${WORK_DIR}/build")
set(reply "${build}/.cmake/api/v1/reply")
file(WRITE "${build}/.cmake/api/v1/query/codemodel-v2" "")
execute_process(
    COMMAND ${environment} "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${build}" ${arguments}
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "A project that embeds RachisRHI with add_subdirectory failed to configure (${status}).")
endif()

execute_process(
    COMMAND "${CTEST_COMMAND}" --test-dir "${build}" -C "${CONFIG}" -N
    RESULT_VARIABLE status
    OUTPUT_VARIABLE listing
    ERROR_VARIABLE listing)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "ctest -N failed in the embedding project (${status}):\n${listing}")
endif()
if(NOT listing MATCHES "Total Tests: 0")
    message(FATAL_ERROR
        "The embedding project includes CTest, but an embedded RachisRHI must register no tests:\n${listing}")
endif()

file(GLOB index "${reply}/index-*.json")
list(LENGTH index index_count)
if(NOT index_count EQUAL 1)
    message(FATAL_ERROR "Expected one CMake File API index in ${reply}, found ${index_count}.")
endif()
file(READ "${index}" index)
string(JSON codemodel_file GET "${index}" reply codemodel-v2 jsonFile)
file(READ "${reply}/${codemodel_file}" codemodel)
string(JSON targets GET "${codemodel}" configurations 0 targets)
string(JSON target_count LENGTH "${targets}")
set(embedded_dir "${build}/rachis")
set(generated_dir "${embedded_dir}/Generated")
set(shaders)
set(tool_artifact)
math(EXPR last_target "${target_count} - 1")
foreach(target_index RANGE ${last_target})
    string(JSON target_file GET "${targets}" ${target_index} jsonFile)
    file(READ "${reply}/${target_file}" target)
    string(JSON target_name GET "${target}" name)
    if(target_name STREQUAL "RachisShaderBinaryTool")
        string(JSON tool_artifact GET "${target}" artifacts 0 path)
        cmake_path(ABSOLUTE_PATH tool_artifact BASE_DIRECTORY "${build}" NORMALIZE)
    endif()
    string(JSON sources ERROR_VARIABLE no_sources GET "${target}" sources)
    if(no_sources)
        continue()
    endif()
    string(JSON source_count LENGTH "${sources}")
    math(EXPR last_source "${source_count} - 1")
    foreach(source_index RANGE ${last_source})
        string(JSON path GET "${sources}" ${source_index} path)
        cmake_path(ABSOLUTE_PATH path BASE_DIRECTORY "${SOURCE_DIR}" NORMALIZE)
        cmake_path(IS_PREFIX generated_dir "${path}" NORMALIZE generated)
        if(generated)
            cmake_path(GET path FILENAME name)
            string(REGEX REPLACE "\\.rule$" "" name "${name}")
            list(APPEND shaders "${name}")
        endif()
    endforeach()
endforeach()
if(NOT shaders)
    message(FATAL_ERROR "The File API lists no shader that the embedded RachisRHI compiles into ${generated_dir}.")
endif()
list(REMOVE_DUPLICATES shaders)
set(unexpected ${shaders})
list(FILTER unexpected EXCLUDE REGEX "^(Rachis)?GenerateMipmaps([123]D)?\\.|^(Rachis)?CountedDispatchArguments\\.|^RachisResolveDepth(Vertex|Pixel)\\.(h|dxil)$")
if(unexpected)
    list(JOIN unexpected "\n  " unexpected)
    message(FATAL_ERROR
        "An embedded RachisRHI must compile only the shaders its libraries embed, but it also compiles:\n"
        "  ${unexpected}")
endif()
if(NOT SHADER_BINARY_TOOL)
    if(NOT tool_artifact)
        message(FATAL_ERROR "The File API lists no RachisShaderBinaryTool that the embedded RachisRHI builds.")
    endif()
    cmake_path(IS_PREFIX embedded_dir "${tool_artifact}" NORMALIZE tool_inside)
    if(NOT tool_inside)
        message(FATAL_ERROR
            "The embedded RachisRHI writes RachisShaderBinaryTool to ${tool_artifact}, outside its own "
            "binary directory ${embedded_dir}.")
    endif()
endif()

set(cross_arguments ${arguments})
list(FILTER cross_arguments EXCLUDE REGEX "^-DRACHIS_SHADER_BINARY_TOOL=")
list(APPEND cross_arguments "-DCMAKE_SYSTEM_NAME=${SYSTEM_NAME}" "-DCMAKE_SYSTEM_PROCESSOR=${SYSTEM_PROCESSOR}")
execute_process(
    COMMAND ${environment} "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${WORK_DIR}/cross" ${cross_arguments}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
string(REGEX REPLACE "[ \t\r\n]+" " " flat_output "${output}")
if(status EQUAL 0)
    message(FATAL_ERROR
        "A project that embeds RachisRHI and calls rachis_add_shader configured for cross-compiling without "
        "RACHIS_SHADER_BINARY_TOOL, so its build would run a RachisShaderBinaryTool that cannot run on the host.")
endif()
if(NOT flat_output MATCHES "Set RACHIS_SHADER_BINARY_TOOL to a host build of it when cross-compiling"
        OR (NOT BACKENDS STREQUAL "metal" AND NOT flat_output MATCHES "\\(rachis_add_shader\\)"))
    message(FATAL_ERROR
        "The embedding project failed to configure for cross-compiling, but not where the build would first "
        "run RachisShaderBinaryTool:\n${output}")
endif()
message(STATUS
    "The embedding project configured, ctest -N lists no RachisRHI tests, RachisRHI compiles only its "
    "internal shaders and keeps RachisShaderBinaryTool in its own binary directory, and cross-compiling "
    "without RACHIS_SHADER_BINARY_TOOL fails to configure.")
