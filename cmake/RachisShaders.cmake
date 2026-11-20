# Compiles HLSL, Slang, or GLSL shaders for the backends of a Rachis build and embeds them in C++
# headers.
#
#   rachis_add_shader(<target>
#       NAME <identifier>
#       SOURCE <file>
#       [LANGUAGE <HLSL|SLANG|GLSL>]
#       ENTRY <entry point>
#       STAGE <vertex|pixel|compute>
#       [SHADER_MODEL <6_0|6_1|6_2|6_3|6_4|6_5|6_6>]
#       [DEFINES <NAME=VALUE>...]
#       [REFLECT])
#
# LANGUAGE defaults to the one SOURCE's extension names: .hlsl is HLSL, .slang is Slang, and
# .glsl, .vert, .frag, and .comp are GLSL. Any other extension needs LANGUAGE. ENTRY is required
# for HLSL and Slang. A GLSL shader's entry function is main, and ENTRY, which defaults to main,
# names its SPIR-V entry point.
#
# SHADER_MODEL, 6_0 by default, is the HLSL shader model an HLSL or Slang shader is compiled for on
# every backend: the dxc profile (cs_6_6 for SHADER_MODEL 6_6 and STAGE compute) and the slangc
# -profile (sm_6_6). Raise it for shaders that use a feature of a later shader model, such as
# 64-bit atomics (6_6). GLSL shaders ignore it.
#
# For every backend that Rachis::RHI was built with, the entry point is compiled with the
# toolchain that backend and language expect:
#
#                HLSL                         Slang              GLSL
#   Direct3D 12  dxc DXIL                     slangc DXIL        not compiled
#   Vulkan       Vulkan SDK dxc SPIR-V        slangc SPIR-V      glslang SPIR-V
#   Metal        slangc and RachisShaderBinaryTool PNMS          not compiled
#
# Every backend defines RACHIS_D3D12=1, RACHIS_VULKAN=1, or RACHIS_METAL=1, and Vulkan SPIR-V is
# for vulkan1.3, checked with spirv-val. HLSL and Slang SPIR-V also define __spirv__, which dxc
# defines itself. GLSL is Vulkan-only: on the other backends its bytecode is empty, and
# Rachis::RHI needs Vulkan. HLSL and Slang compile with the RachisRHI include directory on the
# include path, so a shader can include "Rachis/DrawIndex.hlsli".
#
# The results are embedded in "<identifier>.h", which <target> can include. It declares
#
#   std::span<const std::byte> rachis_shaders::<identifier>(rachis::Backend backend);
#
# which returns the bytecode for ShaderModuleDesc::bytecode, or an empty span for a backend
# the package was built without or the language does not compile for.
#
# With REFLECT, the header also holds the shader's interface. For HLSL and Slang, slangc reflects
# it from the same source once per backend, with that backend's defines, so REFLECT needs slangc
# whatever the backends are. For GLSL, RachisShaderBinaryTool reads it from the SPIR-V glslang
# emits. rachis_shaders::reflection::<identifier> has
#
#   ShaderInterfaceDesc shaderInterface(Backend)      bindings, push constants, vertex inputs,
#                                                     workgroup size, and draw index
#   span<const BindingLayoutEntry> bindingLayout(Backend, set)
#                                                     the set's entries, visible to the stage
#   span<const BindingPoolSize> bindingPoolSizes(Backend, set)
#                                                     the pool sizes of one binding set
#   span<const PushConstantRange> pushConstantRanges(Backend)
#   uint32_t bindingSetCount(Backend)
#   bool drawIndex(Backend)                           whether the shader reads the draw index, for
#                                                     PipelineLayoutDesc::drawIndex
#   ShaderModuleDesc moduleDesc(Backend, label)       stage, bytecode, entry point, interface
#
# and the same data per backend in its d3d12, vulkan, and metal namespaces. Resources declared as
# entry-point parameters are part of the interface. Slang reflects their bindings from the start of
# the entry point's own parameters, so keep global resources out of the set they start in; on
# Direct3D 12 their register space comes from the DXIL. Each backend namespace
# also has a C++ struct for every constant buffer, push-constant block, and structured-buffer
# element the shader uses, laid out as that backend lays it out, with static_assert checks of
# every member's offset and of the size and alignment. Direct3D 12 and Metal layouts come from
# slangc's reflection of that target; Vulkan layouts come from the SPIR-V the Vulkan compiler emits.
# createShaderModule still checks the interface against the bytecode. Call rachis_add_shader in
# the directory that creates <target>.

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/RachisShaderCooking.cmake")

function(_rachis_shaders_embed target name backend binary symbol out_header)
    rachis_ensure_shader_binary_tool(tool_command tool_depends)
    get_filename_component(directory "${binary}" DIRECTORY)
    set(header "${directory}/${name}.${backend}.h")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND
            "${tool_command}"
            embed-header
            --input "${binary}"
            --output "${header}"
            --symbol "${symbol}"
        DEPENDS "${binary}" ${tool_depends}
        COMMENT "Embedding the ${backend} shader ${name} for ${target}"
        VERBATIM
    )
    set("${out_header}" "${header}" PARENT_SCOPE)
endfunction()

# Runs slangc -reflection-json on one backend's compilation of the source; no code is generated. The
# depfile lists every file slangc read, so editing an included file reflects the shader again.
function(_rachis_shaders_reflect target name backend slang_target source entry stage profile stem defines includes
    out_json)
    set(json "${stem}.${backend}.reflection.json")
    set(define_args)
    foreach(define IN LISTS defines)
        if(NOT define STREQUAL "")
            list(APPEND define_args "-D${define}")
        endif()
    endforeach()
    set(include_args)
    foreach(directory IN LISTS includes)
        list(APPEND include_args -I "${directory}")
    endforeach()
    set(target_args -target "${slang_target}")
    if(slang_target STREQUAL "spirv")
        list(APPEND target_args
            -fvk-use-entrypoint-name
            -fvk-b-shift 0 all
            -fvk-t-shift 0 all
            -fvk-s-shift 0 all
            -fvk-u-shift 0 all)
    endif()
    get_filename_component(directory "${stem}" DIRECTORY)
    add_custom_command(
        OUTPUT "${json}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${directory}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DTARGET=${json}"
            "-DDEPFILE=${json}.d"
            -P "${RACHIS_SHADER_REFLECTION_SCRIPT}"
            --
            "${RACHIS_SLANGC_EXECUTABLE}"
            "${source}"
            ${target_args}
            -no-codegen
            -profile "${profile}"
            -matrix-layout-column-major
            -Wno-39029
            -Wno-39001
            -Wno-40100
            ${define_args}
            ${include_args}
            -entry "${entry}"
            -stage "${stage}"
            -reflection-json "${json}"
        DEPENDS "${source}" "${RACHIS_SHADER_REFLECTION_SCRIPT}"
        DEPFILE "${json}.d"
        COMMENT "Reflecting the ${backend} interface of ${name} for ${target}"
        VERBATIM
    )
    set("${out_json}" "${json}" PARENT_SCOPE)
endfunction()

# The language of a shader source: LANGUAGE when given, otherwise the one its extension names.
function(_rachis_shaders_language source language out_language)
    if(NOT "${language}" STREQUAL "")
        string(TOUPPER "${language}" upper)
        if(NOT upper MATCHES "^(HLSL|SLANG|GLSL)$")
            message(FATAL_ERROR "rachis_add_shader: LANGUAGE must be HLSL, SLANG, or GLSL (got '${language}').")
        endif()
        set("${out_language}" "${upper}" PARENT_SCOPE)
        return()
    endif()
    cmake_path(GET source EXTENSION LAST_ONLY extension)
    string(TOLOWER "${extension}" extension)
    if(extension STREQUAL ".hlsl")
        set("${out_language}" HLSL PARENT_SCOPE)
    elseif(extension STREQUAL ".slang")
        set("${out_language}" SLANG PARENT_SCOPE)
    elseif(extension MATCHES "^\\.(glsl|vert|frag|comp)$")
        set("${out_language}" GLSL PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "rachis_add_shader: the extension of '${source}' names no shader language. Pass LANGUAGE HLSL, "
            "SLANG, or GLSL, or name the file .hlsl, .slang, or .glsl (or .vert, .frag, .comp).")
    endif()
endfunction()

function(rachis_add_shader target)
    cmake_parse_arguments(PARSE_ARGV 1 SHADER "REFLECT" "NAME;SOURCE;LANGUAGE;ENTRY;STAGE;SHADER_MODEL" "DEFINES")
    if(SHADER_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "rachis_add_shader: unknown arguments '${SHADER_UNPARSED_ARGUMENTS}'.")
    endif()
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "rachis_add_shader: '${target}' is not a target.")
    endif()
    foreach(required IN ITEMS NAME SOURCE STAGE)
        if(NOT SHADER_${required})
            message(FATAL_ERROR "rachis_add_shader: ${required} is required.")
        endif()
    endforeach()
    _rachis_shaders_language("${SHADER_SOURCE}" "${SHADER_LANGUAGE}" language)
    if(NOT SHADER_ENTRY)
        if(NOT language STREQUAL "GLSL")
            message(FATAL_ERROR "rachis_add_shader: ENTRY is required.")
        endif()
        set(SHADER_ENTRY main)
    endif()
    if(NOT SHADER_ENTRY MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        message(FATAL_ERROR "rachis_add_shader: ENTRY '${SHADER_ENTRY}' is not an identifier.")
    endif()
    if(NOT SHADER_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        message(FATAL_ERROR "rachis_add_shader: NAME '${SHADER_NAME}' is not a C++ identifier.")
    endif()
    get_property(shader_names GLOBAL PROPERTY RACHIS_SHADER_NAMES)
    if("${target}/${SHADER_NAME}" IN_LIST shader_names)
        message(FATAL_ERROR
            "rachis_add_shader: '${target}' already has a shader named '${SHADER_NAME}'. "
            "NAME must be unique within a target.")
    endif()
    set_property(GLOBAL APPEND PROPERTY RACHIS_SHADER_NAMES "${target}/${SHADER_NAME}")
    if("${SHADER_SHADER_MODEL}" STREQUAL "")
        set(SHADER_SHADER_MODEL 6_0)
    endif()
    if(NOT SHADER_SHADER_MODEL MATCHES "^6_[0-6]$")
        message(FATAL_ERROR
            "rachis_add_shader: SHADER_MODEL must be 6_0, 6_1, 6_2, 6_3, 6_4, 6_5, or 6_6 "
            "(got '${SHADER_SHADER_MODEL}').")
    endif()
    set(slang_profile "sm_${SHADER_SHADER_MODEL}")
    if(SHADER_STAGE STREQUAL "vertex")
        set(dxc_profile "vs_${SHADER_SHADER_MODEL}")
    elseif(SHADER_STAGE STREQUAL "pixel")
        set(dxc_profile "ps_${SHADER_SHADER_MODEL}")
    elseif(SHADER_STAGE STREQUAL "compute")
        set(dxc_profile "cs_${SHADER_SHADER_MODEL}")
    else()
        message(FATAL_ERROR
            "rachis_add_shader: STAGE must be vertex, pixel, or compute (got '${SHADER_STAGE}').")
    endif()
    if(NOT TARGET Rachis::RHI)
        message(FATAL_ERROR "rachis_add_shader requires Rachis::RHI. Call find_package(Rachis) first.")
    endif()

    # Variables the cooking functions read live in the scope of the module that sets them, so
    # load them into this function's scope; the functions themselves are already defined.
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RachisShaderCooking.cmake")

    foreach(backend IN ITEMS D3D12 VULKAN METAL)
        get_target_property(enabled_${backend} Rachis::RHI RACHIS_ENABLE_${backend})
        if(NOT enabled_${backend})
            set(enabled_${backend} OFF)
        endif()
    endforeach()
    # Every compiler run for HLSL and Slang gets the RachisRHI include directory, so a shader can include
    # Rachis/DrawIndex.hlsli.
    get_target_property(shader_includes Rachis::RHI RACHIS_SHADER_INCLUDE_DIR)
    if(NOT shader_includes)
        set(shader_includes)
    endif()
    if(language STREQUAL "GLSL" AND NOT enabled_VULKAN)
        message(FATAL_ERROR
            "rachis_add_shader: GLSL shaders compile for Vulkan only, and Rachis::RHI was built without Vulkan. "
            "Write '${SHADER_NAME}' in HLSL or Slang to run it on Direct3D 12 or Metal.")
    endif()
    # REFLECT reflects an HLSL or Slang interface with slangc on every backend, including those whose compiler
    # is not slangc; a GLSL interface comes from its SPIR-V. Without REFLECT, only the backends' compilers are
    # needed.
    if(SHADER_REFLECT AND NOT language STREQUAL "GLSL")
        if(NOT RACHIS_SLANGC_EXECUTABLE
            AND DEFINED ENV{SLANGC}
            AND NOT "$ENV{SLANGC}" STREQUAL ""
            AND EXISTS "$ENV{SLANGC}"
            AND NOT IS_DIRECTORY "$ENV{SLANGC}")
            file(TO_CMAKE_PATH "$ENV{SLANGC}" slangc_from_environment)
            set(RACHIS_SLANGC_EXECUTABLE "${slangc_from_environment}" CACHE FILEPATH "Path to a program.")
        endif()
        if(NOT RACHIS_SLANGC_EXECUTABLE)
            rachis_shader_tool_search_roots(slangc_roots)
            find_program(RACHIS_SLANGC_EXECUTABLE NAMES slangc PATHS ${slangc_roots})
        endif()
        if(NOT RACHIS_SLANGC_EXECUTABLE)
            message(FATAL_ERROR
                "rachis_add_shader: REFLECT needs slangc to reflect the interface of '${SHADER_NAME}', and slangc "
                "was not found. Install Slang, set SLANGC, or put slangc on PATH, or leave out REFLECT.")
        endif()
    endif()
    # GLSL compiles for Vulkan only.
    if(language STREQUAL "GLSL")
        set(enabled_D3D12 OFF)
        set(enabled_METAL OFF)
    endif()
    set(compiler_targets)
    if(enabled_D3D12)
        if(language STREQUAL "SLANG")
            list(APPEND compiler_targets slang-dxil)
        else()
            list(APPEND compiler_targets dxil)
        endif()
    endif()
    if(enabled_VULKAN)
        if(language STREQUAL "SLANG")
            list(APPEND compiler_targets slang-spirv)
        elseif(language STREQUAL "GLSL")
            list(APPEND compiler_targets glsl-spirv)
        else()
            list(APPEND compiler_targets spirv)
        endif()
    endif()
    if(enabled_METAL)
        list(APPEND compiler_targets metallib)
    endif()
    rachis_find_shader_compilers(${compiler_targets})

    cmake_path(ABSOLUTE_PATH SHADER_SOURCE NORMALIZE)
    set(directory "${CMAKE_CURRENT_BINARY_DIR}/rachis_shaders/${target}")
    set(binary_stem "${directory}/${SHADER_NAME}")
    string(MAKE_C_IDENTIFIER "${target}" target_tag)
    string(REGEX REPLACE "_+" "_" target_tag "${target_tag}")
    string(REGEX REPLACE "^_|_$" "" target_tag "${target_tag}")
    string(SHA256 tag "${PROJECT_NAME}/${target}/${SHADER_NAME}")
    string(SUBSTRING "${tag}" 0 12 tag)
    if(target_tag)
        string(PREPEND tag "${target_tag}_")
    endif()
    set(symbol_stem "rachis_shader_${SHADER_NAME}_${tag}")
    set(headers)
    set(includes)
    set(cases)
    set(reflections)
    set(layouts)
    set(bytecodes)

    set(spirv_reflections)
    if(enabled_D3D12)
        if(language STREQUAL "SLANG")
            rachis_cook_slang_dxil(
                SOURCE "${SHADER_SOURCE}"
                ENTRY "${SHADER_ENTRY}"
                STAGE "${SHADER_STAGE}"
                PROFILE "${slang_profile}"
                OUTPUT "${binary_stem}.dxil"
                DEFINES RACHIS_D3D12=1 ${SHADER_DEFINES}
                INCLUDE_DIRECTORIES ${shader_includes}
                COMMENT "Compiling DXIL ${SHADER_NAME} for ${target}"
            )
        else()
            rachis_cook_dxc_dxil(
                SOURCE "${SHADER_SOURCE}"
                ENTRY "${SHADER_ENTRY}"
                PROFILE "${dxc_profile}"
                OUTPUT "${binary_stem}.dxil"
                DEFINES RACHIS_D3D12=1 ${SHADER_DEFINES}
                INCLUDE_DIRECTORIES ${shader_includes}
                COMMENT "Compiling DXIL ${SHADER_NAME} for ${target}"
            )
        endif()
        _rachis_shaders_embed("${target}" "${SHADER_NAME}" d3d12 "${binary_stem}.dxil" "${symbol_stem}_dxil" header)
        if(SHADER_REFLECT)
            _rachis_shaders_reflect("${target}" "${SHADER_NAME}" d3d12 hlsl "${SHADER_SOURCE}" "${SHADER_ENTRY}"
                "${SHADER_STAGE}" "${slang_profile}" "${binary_stem}" "RACHIS_D3D12=1;${SHADER_DEFINES}" "${shader_includes}"
                reflection)
            list(APPEND reflections "d3d12=${reflection}")
            # slangc reflects every resource the source declares; the DXIL tells whether this entry point reads
            # the draw index.
            list(APPEND bytecodes "d3d12=${binary_stem}.dxil")
        endif()
        list(APPEND headers "${header}")
        string(APPEND includes "#include \"${SHADER_NAME}.d3d12.h\"\n")
        string(APPEND cases
            "    case rachis::Backend::d3d12:\n"
            "        return std::as_bytes(std::span{${symbol_stem}_dxil});\n")
    endif()

    if(enabled_VULKAN)
        if(language STREQUAL "SLANG")
            # dxc -spirv defines __spirv__, and slangc does not; define it, so a source behaves the same as HLSL
            # and as Slang.
            rachis_cook_slang_spirv(
                SOURCE "${SHADER_SOURCE}"
                ENTRY "${SHADER_ENTRY}"
                STAGE "${SHADER_STAGE}"
                PROFILE "${slang_profile}"
                OUTPUT "${binary_stem}.spv"
                DEFINES RACHIS_VULKAN=1 __spirv__=1 ${SHADER_DEFINES}
                INCLUDE_DIRECTORIES ${shader_includes}
                COMMENT "Compiling SPIR-V ${SHADER_NAME} for ${target}"
            )
        elseif(language STREQUAL "GLSL")
            rachis_cook_glslang_spirv(
                SOURCE "${SHADER_SOURCE}"
                ENTRY "${SHADER_ENTRY}"
                STAGE "${SHADER_STAGE}"
                OUTPUT "${binary_stem}.spv"
                DEFINES RACHIS_VULKAN=1 ${SHADER_DEFINES}
                COMMENT "Compiling SPIR-V ${SHADER_NAME} for ${target}"
            )
        else()
            rachis_cook_dxc_spirv(
                SOURCE "${SHADER_SOURCE}"
                ENTRY "${SHADER_ENTRY}"
                PROFILE "${dxc_profile}"
                OUTPUT "${binary_stem}.spv"
                DEFINES RACHIS_VULKAN=1 ${SHADER_DEFINES}
                INCLUDE_DIRECTORIES ${shader_includes}
                COMMENT "Compiling SPIR-V ${SHADER_NAME} for ${target}"
            )
        endif()
        _rachis_shaders_embed("${target}" "${SHADER_NAME}" vulkan "${binary_stem}.spv" "${symbol_stem}_spirv" header)
        if(SHADER_REFLECT AND language STREQUAL "GLSL")
            # Slang's GLSL reflection misses the workgroup size and storage-buffer access, so a GLSL interface
            # and its data layouts come from the SPIR-V glslang emits.
            list(APPEND spirv_reflections "vulkan=${binary_stem}.spv")
        elseif(SHADER_REFLECT)
            # dxc -spirv defines __spirv__; define it for slangc too, so both see the same source.
            _rachis_shaders_reflect("${target}" "${SHADER_NAME}" vulkan spirv "${SHADER_SOURCE}" "${SHADER_ENTRY}"
                "${SHADER_STAGE}" "${slang_profile}" "${binary_stem}" "RACHIS_VULKAN=1;__spirv__=1;${SHADER_DEFINES}"
                "${shader_includes}"
                reflection)
            list(APPEND reflections "vulkan=${reflection}")
            # dxc's relaxed std140 and std430 rules differ from Slang's, so Vulkan data layouts come from the
            # SPIR-V the compiler emitted.
            list(APPEND layouts "vulkan=${binary_stem}.spv")
            list(APPEND bytecodes "vulkan=${binary_stem}.spv")
        endif()
        list(APPEND headers "${header}")
        string(APPEND includes "#include \"${SHADER_NAME}.vulkan.h\"\n")
        string(APPEND cases
            "    case rachis::Backend::vulkan:\n"
            "        return std::as_bytes(std::span{${symbol_stem}_spirv});\n")
    endif()

    if(enabled_METAL)
        rachis_cook_slang_metallib(
            SOURCE "${SHADER_SOURCE}"
            ENTRY "${SHADER_ENTRY}"
            STAGE "${SHADER_STAGE}"
            PROFILE "${slang_profile}"
            OUTPUT "${binary_stem}.metallib"
            DEFINES ${SHADER_DEFINES}
            INCLUDE_DIRECTORIES ${shader_includes}
            COMMENT "Compiling the Metal library ${SHADER_NAME} for ${target}"
        )
        _rachis_shaders_embed("${target}" "${SHADER_NAME}" metal "${binary_stem}.pnms" "${symbol_stem}_pnms" header)
        if(SHADER_REFLECT)
            # Metal's logical sets and bindings follow the SPIR-V layout, as RachisShaderBinaryTool plans them.
            _rachis_shaders_reflect("${target}" "${SHADER_NAME}" metal spirv "${SHADER_SOURCE}" "${SHADER_ENTRY}"
                "${SHADER_STAGE}" "${slang_profile}" "${binary_stem}" "RACHIS_METAL=1;${SHADER_DEFINES}" "${shader_includes}"
                reflection)
            list(APPEND reflections "metal=${reflection}")
            # Slang compiles Metal itself, so its Metal reflection gives the data layouts.
            _rachis_shaders_reflect("${target}" "${SHADER_NAME}" metal-layout metal "${SHADER_SOURCE}"
                "${SHADER_ENTRY}" "${SHADER_STAGE}" "${slang_profile}" "${binary_stem}" "RACHIS_METAL=1;${SHADER_DEFINES}"
                "${shader_includes}"
                layout)
            list(APPEND layouts "metal=${layout}")
            list(APPEND bytecodes "metal=${binary_stem}.pnms")
        endif()
        list(APPEND headers "${header}")
        string(APPEND includes "#include \"${SHADER_NAME}.metal.h\"\n")
        string(APPEND cases
            "    case rachis::Backend::metal:\n"
            "        return std::as_bytes(std::span{${symbol_stem}_pnms});\n")
    endif()

    if(NOT headers)
        message(FATAL_ERROR "rachis_add_shader: Rachis::RHI enables no backend.")
    endif()

    set(reflection_includes)
    set(reflection_namespace)
    if(SHADER_REFLECT)
        rachis_ensure_shader_binary_tool(tool_command tool_depends)
        set(interface_header "${directory}/${SHADER_NAME}.interface.h")
        set(reflection_args)
        set(reflection_files)
        foreach(item IN LISTS reflections)
            list(APPEND reflection_args --reflection "${item}")
            string(REGEX REPLACE "^[a-z0-9]+=" "" file "${item}")
            list(APPEND reflection_files "${file}")
        endforeach()
        foreach(item IN LISTS layouts)
            list(APPEND reflection_args --layout "${item}")
            string(REGEX REPLACE "^[a-z0-9]+=" "" file "${item}")
            list(APPEND reflection_files "${file}")
        endforeach()
        foreach(item IN LISTS spirv_reflections)
            list(APPEND reflection_args --spirv-reflection "${item}")
            string(REGEX REPLACE "^[a-z0-9]+=" "" file "${item}")
            list(APPEND reflection_files "${file}")
        endforeach()
        foreach(item IN LISTS bytecodes)
            list(APPEND reflection_args --bytecode "${item}")
            string(REGEX REPLACE "^[a-z0-9]+=" "" file "${item}")
            list(APPEND reflection_files "${file}")
        endforeach()
        add_custom_command(
            OUTPUT "${interface_header}"
            COMMAND
                "${tool_command}"
                reflect-interface
                --name "${SHADER_NAME}"
                --stage "${SHADER_STAGE}"
                --entry "${SHADER_ENTRY}"
                --tag "${tag}"
                --output "${interface_header}"
                ${reflection_args}
            DEPENDS ${reflection_files} ${tool_depends}
            COMMENT "Generating the shader interface of ${SHADER_NAME} for ${target}"
            VERBATIM
        )
        list(APPEND headers "${interface_header}")
        set(reflection_includes "#include <string_view>\n\n#include \"${SHADER_NAME}.interface.h\"\n")
        set(reflection_namespace "
namespace rachis_shaders::reflection::${SHADER_NAME}
{
inline namespace t_${tag}
{

// A ShaderModuleDesc with this shader's stage, bytecode, entry point, and interface for the backend.
[[nodiscard]] inline rachis::ShaderModuleDesc moduleDesc(rachis::Backend backend, std::string_view label = {}) noexcept
{
    return rachis::ShaderModuleDesc{
        .stage = stage,
        .bytecode = ::rachis_shaders::${SHADER_NAME}(backend),
        .entryPoint = entryPoint,
        .label = label,
        .shaderInterface = shaderInterface(backend),
    };
}

}
}
")
    endif()

    file(GENERATE OUTPUT "${directory}/${SHADER_NAME}.h" CONTENT
"// Generated by rachis_add_shader from ${SHADER_SOURCE}
// (${language}, entry point ${SHADER_ENTRY}, stage ${SHADER_STAGE}). Do not edit.
#pragma once

#include <Rachis/Device.h>

#include <cstddef>
#include <span>
${reflection_includes}
${includes}
namespace rachis_shaders
{
inline namespace t_${tag}
{

[[nodiscard]] inline std::span<const std::byte> ${SHADER_NAME}(rachis::Backend backend) noexcept
{
    switch (backend)
    {
${cases}    default:
        return {};
    }
}

}
}
${reflection_namespace}")

    target_sources("${target}" PRIVATE ${headers} "${directory}/${SHADER_NAME}.h")
    target_include_directories("${target}" PRIVATE "${directory}")
endfunction()
