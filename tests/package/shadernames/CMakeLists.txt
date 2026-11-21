cmake_minimum_required(VERSION 4.0)
project(RachisShaderNames LANGUAGES CXX)

find_package(Rachis 0.3 CONFIG REQUIRED)
enable_testing()

set(rachis_shader_names_backends)
foreach(backend IN ITEMS D3D12 VULKAN METAL)
    get_target_property(rachis_enable_${backend} Rachis::RHI RACHIS_ENABLE_${backend})
    if(rachis_enable_${backend} MATCHES "-NOTFOUND$")
        message(FATAL_ERROR "Rachis::RHI does not export RACHIS_ENABLE_${backend}.")
    endif()
    if(rachis_enable_${backend})
        string(TOLOWER "${backend}" backend)
        list(APPEND rachis_shader_names_backends "${backend}")
    endif()
endforeach()

function(add_fill_shader target offset)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_link_libraries(${target} PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
    rachis_add_shader(${target}
        NAME FillShader
        SOURCE "${PROJECT_SOURCE_DIR}/Fill.hlsl"
        ENTRY fillMain
        STAGE compute
        DEFINES FILL_OFFSET=${offset})
endfunction()

function(add_shader_names_program name second_type)
    add_library(${name}First STATIC "${PROJECT_SOURCE_DIR}/First.cpp")
    add_fill_shader(${name}First 1)
    add_library(${name}Second ${second_type} "${PROJECT_SOURCE_DIR}/Second.cpp")
    add_fill_shader(${name}Second 2)
    add_executable(${name} "${PROJECT_SOURCE_DIR}/main.cpp")
    add_fill_shader(${name} 3)
    target_link_libraries(${name} PRIVATE ${name}First ${name}Second)
    set(shaders "${CMAKE_CURRENT_BINARY_DIR}/rachis_shaders")
    add_test(NAME ${name}
        COMMAND ${name}
            "${shaders}/${name}"
            "${shaders}/${name}First"
            "${shaders}/${name}Second"
            ${rachis_shader_names_backends})
endfunction()

add_shader_names_program(RachisShaderNames STATIC)
if(NOT WIN32)
    add_shader_names_program(RachisShaderNamesShared SHARED)
endif()
add_subdirectory(optimized)

set(reuse_arguments -G "${CMAKE_GENERATOR}" "-DRachis_DIR=${Rachis_DIR}")
foreach(variable IN ITEMS
        CMAKE_MAKE_PROGRAM
        CMAKE_CXX_COMPILER
        CMAKE_RC_COMPILER
        CMAKE_MT
        CMAKE_OSX_ARCHITECTURES
        CMAKE_OSX_DEPLOYMENT_TARGET
        RACHIS_DXC_EXECUTABLE
        RACHIS_DXC_SPIRV_EXECUTABLE
        RACHIS_SPIRV_VAL_EXECUTABLE
        RACHIS_SLANGC_EXECUTABLE
        RACHIS_SHADER_BINARY_TOOL)
    if(NOT "${${variable}}" STREQUAL "")
        list(APPEND reuse_arguments "-D${variable}=${${variable}}")
    endif()
endforeach()
add_test(NAME RachisShaderNamesReuse
    COMMAND "${CMAKE_COMMAND}"
        -S "${PROJECT_SOURCE_DIR}/reuse"
        -B "${CMAKE_CURRENT_BINARY_DIR}/reuse"
        ${reuse_arguments})
set_tests_properties(RachisShaderNamesReuse PROPERTIES
    PASS_REGULAR_EXPRESSION
        "'RachisShaderNamesReuse' already has a shader named[ \r\n]+'FillShader'\\.[ \r\n]+NAME must be unique[ \r\n]+within a target\\.")

# A source whose extension names no language needs LANGUAGE, and LANGUAGE takes only the languages it compiles.
add_test(NAME RachisShaderLanguageFromExtension
    COMMAND "${CMAKE_COMMAND}"
        -S "${PROJECT_SOURCE_DIR}/language"
        -B "${CMAKE_CURRENT_BINARY_DIR}/ext"
        -DRACHIS_SHADER_LANGUAGE_CASE=extension
        ${reuse_arguments})
set_tests_properties(RachisShaderLanguageFromExtension PROPERTIES
    PASS_REGULAR_EXPRESSION
        "Fill\\.shader'[ \r\n]+names[ \r\n]+no[ \r\n]+shader[ \r\n]+language\\.[ \r\n]+Pass[ \r\n]+LANGUAGE[ \r\n]+HLSL,")
add_test(NAME RachisShaderLanguageUnknown
    COMMAND "${CMAKE_COMMAND}"
        -S "${PROJECT_SOURCE_DIR}/language"
        -B "${CMAKE_CURRENT_BINARY_DIR}/lang"
        -DRACHIS_SHADER_LANGUAGE_CASE=language
        ${reuse_arguments})
set_tests_properties(RachisShaderLanguageUnknown PROPERTIES
    PASS_REGULAR_EXPRESSION
        "LANGUAGE[ \r\n]+must[ \r\n]+be[ \r\n]+HLSL,[ \r\n]+SLANG,[ \r\n]+or[ \r\n]+GLSL[ \r\n]+\\(got[ \r\n]+'MSL'\\)")

# REFLECT needs slangc on every backend: with slangc hidden, configuring a REFLECT shader fails and says so.
list(FILTER reuse_arguments EXCLUDE REGEX "^-DRACHIS_SLANGC_EXECUTABLE=")
add_test(NAME RachisShaderReflectWithoutSlangc
    COMMAND "${CMAKE_COMMAND}" -E env --unset=SLANGC
        "${CMAKE_COMMAND}"
        -S "${PROJECT_SOURCE_DIR}/reflect"
        -B "${CMAKE_CURRENT_BINARY_DIR}/reflect"
        ${reuse_arguments})
set_tests_properties(RachisShaderReflectWithoutSlangc PROPERTIES
    PASS_REGULAR_EXPRESSION
        "rachis_add_shader:[ \r\n]+REFLECT[ \r\n]+needs[ \r\n]+slangc[ \r\n]+to[ \r\n]+reflect[ \r\n]+the[ \r\n]+interface[ \r\n]+of[ \r\n]+'FillShader'")
