set(RACHIS_MIN_SLANGC_VERSION "2026.17.1" CACHE STRING "Minimum supported slangc version")
option(RACHIS_ALLOW_UNPINNED_SLANGC
    "Allow slangc versions older than RACHIS_MIN_SLANGC_VERSION or unparseable version strings" OFF)
option(RACHIS_METAL_EMIT_DEBUG_SOURCE
    "Emit debug .metallib.metal source alongside cooked Metal shaders" OFF)

set(RACHIS_SHADER_COOKING_KEY_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/WriteShaderArtifactKey.cmake")
set(RACHIS_SHADER_DEPFILE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/WriteShaderDepfile.cmake")
set(RACHIS_SHADER_SLANG_DEFAULT_PROFILE sm_6_0)
set(RACHIS_SHADER_SLANG_SPIRV_OPTIONS
    -target spirv
    -matrix-layout-column-major
    -restrictive-capability-check
    -warnings-as-errors all
    -emit-spirv-directly
    -fvk-use-entrypoint-name
    -fvk-b-shift 0 all
    -fvk-t-shift 0 all
    -fvk-s-shift 0 all
    -fvk-u-shift 0 all
)
set(RACHIS_SHADER_SLANG_DXIL_OPTIONS
    -target dxil
    -matrix-layout-column-major
    -restrictive-capability-check
    -warnings-as-errors all
)
set(RACHIS_SHADER_SPIRV_ENVIRONMENT vulkan1.3)
set(RACHIS_SHADER_GLSLANG_SPIRV_OPTIONS
    -V
    --target-env vulkan1.3
)
set(RACHIS_SHADER_DXC_SPIRV_OPTIONS
    -spirv
    -fspv-target-env=vulkan1.3
    -fvk-support-nonzero-base-vertex
    -fvk-support-nonzero-base-instance
)
set(RACHIS_SHADER_METAL_ENVIRONMENT macos26-metal4)
set(RACHIS_SHADER_METAL_MSL macos-metal4)
set(RACHIS_SHADER_SLANG_METALLIB_OPTIONS
    -target metallib
    -matrix-layout-column-major
    -restrictive-capability-check
    -fvk-use-entrypoint-name
    -fvk-b-shift 0 all
    -fvk-t-shift 0 all
    -fvk-s-shift 0 all
    -fvk-u-shift 0 all
)
set(RACHIS_SHADER_SLANG_METAL_SOURCE_OPTIONS
    -target metal
    -profile sm_6_0
    -matrix-layout-column-major
    -restrictive-capability-check
    -fvk-use-entrypoint-name
    -fvk-b-shift 0 all
    -fvk-t-shift 0 all
    -fvk-s-shift 0 all
    -fvk-u-shift 0 all
)

if(COMMAND rachis_cook_slang_spirv)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/RachisShaderBinaryTools.cmake")

function(rachis_prepend_vulkan_sdk_rpath list_var)
    if(NOT DEFINED ENV{VULKAN_SDK})
        return()
    endif()
    set(current "${${list_var}}")
    list(PREPEND current
        "$ENV{VULKAN_SDK}/lib"
        "$ENV{VULKAN_SDK}/macOS/lib"
    )
    set(${list_var} "${current}" PARENT_SCOPE)
endfunction()

function(rachis_shader_tool_search_roots out_var)
    set(roots)
    if(DEFINED ENV{SLANGC} AND NOT "$ENV{SLANGC}" STREQUAL "" AND EXISTS "$ENV{SLANGC}")
        file(TO_CMAKE_PATH "$ENV{SLANGC}" rachis_env_slangc)
        get_filename_component(rachis_env_slangc_dir "${rachis_env_slangc}" DIRECTORY)
        if(rachis_env_slangc_dir)
            list(APPEND roots "${rachis_env_slangc_dir}")
        endif()
    endif()
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND roots
            "$ENV{VULKAN_SDK}/Bin"
            "$ENV{VULKAN_SDK}/bin"
            "$ENV{VULKAN_SDK}/macOS/bin"
        )
    endif()
    file(GLOB rachis_vulkan_sdk_bins "C:/VulkanSDK/*/Bin")
    list(APPEND roots ${rachis_vulkan_sdk_bins})
    list(APPEND roots /usr/bin /usr/local/bin /opt/homebrew/bin)
    set(${out_var} "${roots}" PARENT_SCOPE)
endfunction()

function(rachis_reject_wrong_architecture executable label)
    if(NOT APPLE OR NOT EXISTS "${executable}")
        return()
    endif()
    file(REAL_PATH "${executable}" rachis_resolved_executable)
    execute_process(
        COMMAND file "${rachis_resolved_executable}"
        OUTPUT_VARIABLE rachis_file_output
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(rachis_file_output AND NOT rachis_file_output MATCHES "arm64")
        message(FATAL_ERROR
            "${label} at '${executable}' is not an arm64 binary (${rachis_file_output}). "
            "Install the Apple silicon build; see docs/building.md.")
    endif()
endfunction()

function(rachis_verify_spirv_dxc)
    execute_process(
        COMMAND "${RACHIS_DXC_SPIRV_EXECUTABLE}" -help
        OUTPUT_VARIABLE rachis_dxc_help_out
        ERROR_VARIABLE rachis_dxc_help_err
        RESULT_VARIABLE rachis_dxc_help_result
    )
    set(rachis_dxc_help "${rachis_dxc_help_out}${rachis_dxc_help_err}")
    if(NOT rachis_dxc_help MATCHES "spirv")
        message(FATAL_ERROR
            "A SPIR-V-capable dxc was not found at '${RACHIS_DXC_SPIRV_EXECUTABLE}'. "
            "Use the Vulkan SDK dxc; a DXIL-only compiler cannot cook vulkan1.3 modules.")
    endif()
endfunction()

function(rachis_record_shader_tool_versions)
    if("slangc" IN_LIST ARGN)
        execute_process(
            COMMAND "${RACHIS_SLANGC_EXECUTABLE}" -v
            OUTPUT_VARIABLE rachis_slangc_version
            ERROR_VARIABLE rachis_slangc_version_err
            RESULT_VARIABLE rachis_slangc_version_result
        )
        string(STRIP "${rachis_slangc_version}${rachis_slangc_version_err}" rachis_slangc_version)
        set(RACHIS_SLANGC_VERSION "${rachis_slangc_version}" CACHE INTERNAL "Detected slangc version string")
        message(STATUS "slangc: ${RACHIS_SLANGC_EXECUTABLE} (${rachis_slangc_version})")
    endif()
    if("dxc-spirv" IN_LIST ARGN)
        execute_process(
            COMMAND "${RACHIS_DXC_SPIRV_EXECUTABLE}" --version
            OUTPUT_VARIABLE rachis_dxc_version
            ERROR_VARIABLE rachis_dxc_version_err
        )
        string(STRIP "${rachis_dxc_version}${rachis_dxc_version_err}" rachis_dxc_version)
        message(STATUS "dxc (SPIR-V): ${RACHIS_DXC_SPIRV_EXECUTABLE} (${rachis_dxc_version})")
    endif()
    if("spirv-val" IN_LIST ARGN)
        execute_process(
            COMMAND "${RACHIS_SPIRV_VAL_EXECUTABLE}" --version
            OUTPUT_VARIABLE rachis_spirv_val_version
            ERROR_VARIABLE rachis_spirv_val_version_err
        )
        string(STRIP "${rachis_spirv_val_version}${rachis_spirv_val_version_err}" rachis_spirv_val_version)
        message(STATUS "spirv-val: ${RACHIS_SPIRV_VAL_EXECUTABLE} (${rachis_spirv_val_version})")
    endif()
    if("glslang" IN_LIST ARGN)
        execute_process(
            COMMAND "${RACHIS_GLSLANG_EXECUTABLE}" --version
            OUTPUT_VARIABLE rachis_glslang_version
            ERROR_VARIABLE rachis_glslang_version_err
        )
        string(REGEX MATCH "Glslang Version: [^\r\n]*" rachis_glslang_version
            "${rachis_glslang_version}${rachis_glslang_version_err}")
        message(STATUS "glslang: ${RACHIS_GLSLANG_EXECUTABLE} (${rachis_glslang_version})")
    endif()
endfunction()

function(rachis_find_shader_compilers)
    set(rachis_tools)
    foreach(rachis_target IN LISTS ARGN)
        if(rachis_target STREQUAL "dxil")
            list(APPEND rachis_tools dxc)
        elseif(rachis_target STREQUAL "spirv")
            list(APPEND rachis_tools dxc-spirv spirv-val)
        elseif(rachis_target STREQUAL "slang-dxil")
            list(APPEND rachis_tools slangc)
        elseif(rachis_target STREQUAL "slang-spirv")
            list(APPEND rachis_tools slangc spirv-val)
        elseif(rachis_target STREQUAL "glsl-spirv")
            list(APPEND rachis_tools glslang spirv-val)
        elseif(rachis_target STREQUAL "metallib")
            list(APPEND rachis_tools slangc slangc-version apple-sdk)
        else()
            message(FATAL_ERROR
                "rachis_find_shader_compilers: '${rachis_target}' is not dxil, spirv, slang-dxil, slang-spirv, "
                "glsl-spirv, or metallib.")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES rachis_tools)
    get_property(rachis_checked_tools GLOBAL PROPERTY RACHIS_SHADER_TOOLS_CHECKED)
    if(rachis_checked_tools)
        list(REMOVE_ITEM rachis_tools ${rachis_checked_tools})
    endif()
    if(NOT rachis_tools)
        return()
    endif()

    rachis_shader_tool_search_roots(rachis_tool_roots)
    if("slangc" IN_LIST rachis_tools)
        if(NOT RACHIS_SLANGC_EXECUTABLE
            AND DEFINED ENV{SLANGC}
            AND NOT "$ENV{SLANGC}" STREQUAL ""
            AND EXISTS "$ENV{SLANGC}"
            AND NOT IS_DIRECTORY "$ENV{SLANGC}")
            file(TO_CMAKE_PATH "$ENV{SLANGC}" rachis_env_slangc_executable)
            set(RACHIS_SLANGC_EXECUTABLE "${rachis_env_slangc_executable}"
                CACHE FILEPATH "Path to a program.")
        endif()
        if(NOT RACHIS_SLANGC_EXECUTABLE)
            find_program(RACHIS_SLANGC_EXECUTABLE NAMES slangc PATHS ${rachis_tool_roots})
        endif()
        if(NOT RACHIS_SLANGC_EXECUTABLE)
            if(APPLE)
                message(FATAL_ERROR
                    "slangc was not found. Install Slang ${RACHIS_MIN_SLANGC_VERSION}+ for Apple silicon as documented in "
                    "docs/building.md, set SLANGC, or put slangc on PATH.")
            endif()
            if("metallib" IN_LIST ARGN)
                message(FATAL_ERROR "slangc was not found on PATH. Install Slang before cooking Metal metallib.")
            endif()
            if("slang-dxil" IN_LIST ARGN)
                message(FATAL_ERROR
                    "slangc was not found on PATH. Install Slang, set SLANGC, or put slangc on PATH before "
                    "compiling Slang shaders.")
            endif()
            message(FATAL_ERROR "slangc was not found on PATH. Install Slang before cooking Vulkan SPIR-V.")
        endif()
        rachis_reject_wrong_architecture("${RACHIS_SLANGC_EXECUTABLE}" "slangc")
    endif()
    if("glslang" IN_LIST rachis_tools)
        if(NOT RACHIS_GLSLANG_EXECUTABLE)
            find_program(RACHIS_GLSLANG_EXECUTABLE NAMES glslangValidator glslang PATHS ${rachis_tool_roots})
        endif()
        if(NOT RACHIS_GLSLANG_EXECUTABLE)
            message(FATAL_ERROR
                "glslang was not found. Install glslang (glslangValidator, part of the Vulkan SDK), put it on "
                "PATH, or set RACHIS_GLSLANG_EXECUTABLE before compiling GLSL shaders.")
        endif()
        rachis_reject_wrong_architecture("${RACHIS_GLSLANG_EXECUTABLE}" "glslang")
    endif()
    if("dxc" IN_LIST rachis_tools)
        if(NOT RACHIS_DXC_EXECUTABLE)
            find_program(RACHIS_DXC_EXECUTABLE NAMES dxc)
        endif()
        if(NOT RACHIS_DXC_EXECUTABLE)
            message(FATAL_ERROR
                "dxc was not found on PATH. Install the DirectX Shader Compiler "
                "before configuring a D3D12 RHI backend.")
        endif()
    endif()
    if("dxc-spirv" IN_LIST rachis_tools)
        if(RACHIS_DXC_SPIRV_EXECUTABLE MATCHES "Windows Kits")
            unset(RACHIS_DXC_SPIRV_EXECUTABLE CACHE)
        endif()
        if(NOT RACHIS_DXC_SPIRV_EXECUTABLE)
            if(rachis_tool_roots)
                find_program(RACHIS_DXC_SPIRV_EXECUTABLE NAMES dxc PATHS ${rachis_tool_roots} NO_DEFAULT_PATH)
            endif()
            if(NOT RACHIS_DXC_SPIRV_EXECUTABLE)
                find_program(RACHIS_DXC_SPIRV_EXECUTABLE NAMES dxc)
            endif()
        endif()
        if(NOT RACHIS_DXC_SPIRV_EXECUTABLE)
            if(APPLE)
                message(FATAL_ERROR
                    "A SPIR-V-capable dxc was not found. Install LunarG Vulkan SDK 1.4.357.1, export "
                    "VULKAN_SDK, and see docs/building.md.")
            endif()
            message(FATAL_ERROR
                "A SPIR-V-capable dxc was not found. Use the Vulkan SDK dxc; the Windows SDK "
                "compiler cannot emit SPIR-V.")
        endif()
        rachis_reject_wrong_architecture("${RACHIS_DXC_SPIRV_EXECUTABLE}" "dxc")
        rachis_verify_spirv_dxc()
    endif()
    if("spirv-val" IN_LIST rachis_tools)
        if(NOT RACHIS_SPIRV_VAL_EXECUTABLE)
            find_program(RACHIS_SPIRV_VAL_EXECUTABLE NAMES spirv-val PATHS ${rachis_tool_roots})
            if(NOT RACHIS_SPIRV_VAL_EXECUTABLE AND DEFINED ENV{VULKAN_SDK})
                find_program(RACHIS_SPIRV_VAL_EXECUTABLE NAMES spirv-val
                    PATHS
                        "$ENV{VULKAN_SDK}/Bin"
                        "$ENV{VULKAN_SDK}/bin"
                        "$ENV{VULKAN_SDK}/macOS/bin"
                    NO_DEFAULT_PATH)
            endif()
        endif()
        if(NOT RACHIS_SPIRV_VAL_EXECUTABLE)
            message(FATAL_ERROR
                "spirv-val was not found on PATH. Install SPIRV-Tools from the Vulkan SDK so cooked "
                "modules are validated for vulkan1.3.")
        endif()
        rachis_reject_wrong_architecture("${RACHIS_SPIRV_VAL_EXECUTABLE}" "spirv-val")
    endif()
    rachis_record_shader_tool_versions(${rachis_tools})

    if("apple-sdk" IN_LIST rachis_tools)
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-version
            OUTPUT_VARIABLE rachis_apple_sdk_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE rachis_apple_sdk_result
        )
        if(NOT rachis_apple_sdk_result EQUAL 0 OR NOT rachis_apple_sdk_version)
            message(FATAL_ERROR "xcrun --sdk macosx --show-sdk-version failed. Install Xcode 26+ command-line tools.")
        endif()
        set(RACHIS_APPLE_SDK_VERSION "${rachis_apple_sdk_version}" CACHE INTERNAL "MacOSX SDK version that cooks Metal shaders")
    endif()

    if("slangc-version" IN_LIST rachis_tools AND NOT RACHIS_ALLOW_UNPINNED_SLANGC)
        if(RACHIS_SLANGC_VERSION MATCHES "([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
            set(rachis_slangc_semver "${CMAKE_MATCH_1}")
        else()
            set(rachis_slangc_semver "")
        endif()
        if(NOT rachis_slangc_semver
            OR NOT rachis_slangc_semver VERSION_GREATER_EQUAL "${RACHIS_MIN_SLANGC_VERSION}")
            message(FATAL_ERROR
                "Metal requires Slang ${RACHIS_MIN_SLANGC_VERSION}+ (got '${RACHIS_SLANGC_VERSION}' at "
                "'${RACHIS_SLANGC_EXECUTABLE}'). Set RACHIS_ALLOW_UNPINNED_SLANGC=ON to override.")
        endif()
    endif()
    set_property(GLOBAL APPEND PROPERTY RACHIS_SHADER_TOOLS_CHECKED ${rachis_tools})
endfunction()

function(rachis_shader_depfile_command out_command format output)
    set("${out_command}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DFORMAT=${format}"
            "-DINPUT=${output}.${format}.d"
            "-DOUTPUT=${output}.d"
            "-DTARGET=${output}"
            -P "${RACHIS_SHADER_DEPFILE_SCRIPT}"
        PARENT_SCOPE)
endfunction()

function(rachis_cook_dxc_dxil)
    cmake_parse_arguments(COOK "" "SOURCE;ENTRY;PROFILE;OUTPUT;HEADER;SYMBOL;COMMENT"
        "DEFINES;DEPENDS;OPTIONS;INCLUDE_DIRECTORIES" ${ARGN})
    if(NOT COOK_SOURCE OR NOT COOK_PROFILE OR NOT COOK_OUTPUT)
        message(FATAL_ERROR "rachis_cook_dxc_dxil requires SOURCE, PROFILE, and OUTPUT")
    endif()
    if(NOT RACHIS_DXC_EXECUTABLE)
        message(FATAL_ERROR "rachis_cook_dxc_dxil requires RACHIS_DXC_EXECUTABLE")
    endif()
    set(compile_args -T "${COOK_PROFILE}")
    if(COOK_ENTRY)
        list(APPEND compile_args -E "${COOK_ENTRY}")
    endif()
    list(APPEND compile_args ${COOK_OPTIONS})
    foreach(directory IN LISTS COOK_INCLUDE_DIRECTORIES)
        list(APPEND compile_args -I "${directory}")
    endforeach()
    foreach(define IN LISTS COOK_DEFINES)
        list(APPEND compile_args "-D${define}")
    endforeach()
    set(header_args)
    if(COOK_HEADER)
        if(NOT COOK_SYMBOL)
            message(FATAL_ERROR "rachis_cook_dxc_dxil HEADER requires SYMBOL")
        endif()
        list(APPEND header_args -Fh "${COOK_HEADER}" -Vn "${COOK_SYMBOL}")
    endif()
    cmake_path(GET COOK_OUTPUT PARENT_PATH output_directory)
    rachis_shader_depfile_command(depfile_command dxc "${COOK_OUTPUT}")
    add_custom_command(
        OUTPUT "${COOK_OUTPUT}" ${COOK_HEADER}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND
            "${RACHIS_DXC_EXECUTABLE}"
            ${compile_args}
            -Fo "${COOK_OUTPUT}"
            ${header_args}
            "${COOK_SOURCE}"
        COMMAND
            "${RACHIS_DXC_EXECUTABLE}"
            ${compile_args}
            -MF "${COOK_OUTPUT}.dxc.d"
            "${COOK_SOURCE}"
        ${depfile_command}
        DEPENDS
            "${COOK_SOURCE}"
            "${RACHIS_SHADER_DEPFILE_SCRIPT}"
            ${COOK_DEPENDS}
        DEPFILE "${COOK_OUTPUT}.d"
        COMMENT "${COOK_COMMENT}"
        VERBATIM
    )
endfunction()

function(rachis_cook_slang_spirv)
    cmake_parse_arguments(COOK "" "SOURCE;ENTRY;STAGE;PROFILE;OUTPUT;COMMENT" "DEFINES;DEPENDS;INCLUDE_DIRECTORIES"
        ${ARGN})
    if(NOT COOK_SOURCE OR NOT COOK_ENTRY OR NOT COOK_OUTPUT)
        message(FATAL_ERROR "rachis_cook_slang_spirv requires SOURCE, ENTRY, and OUTPUT")
    endif()
    if(NOT COOK_PROFILE)
        set(COOK_PROFILE "${RACHIS_SHADER_SLANG_DEFAULT_PROFILE}")
    endif()
    set(key_path "${COOK_OUTPUT}.key")
    set(define_args)
    foreach(define IN LISTS COOK_DEFINES)
        list(APPEND define_args "-D${define}")
    endforeach()
    list(JOIN COOK_DEFINES "," define_text)
    set(stage_args)
    if(COOK_STAGE)
        set(stage_args -stage "${COOK_STAGE}")
    endif()
    set(compile_options ${RACHIS_SHADER_SLANG_SPIRV_OPTIONS} -profile "${COOK_PROFILE}" ${stage_args})
    foreach(directory IN LISTS COOK_INCLUDE_DIRECTORIES)
        list(APPEND compile_options -I "${directory}")
    endforeach()
    list(JOIN compile_options " " option_text)
    cmake_path(GET COOK_OUTPUT PARENT_PATH output_directory)
    rachis_shader_depfile_command(depfile_command slangc "${COOK_OUTPUT}")
    add_custom_command(
        OUTPUT "${COOK_OUTPUT}" "${key_path}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND
            "${RACHIS_SLANGC_EXECUTABLE}"
            "${COOK_SOURCE}"
            ${compile_options}
            ${define_args}
            -entry "${COOK_ENTRY}"
            -o "${COOK_OUTPUT}"
            -depfile "${COOK_OUTPUT}.slangc.d"
        COMMAND
            "${RACHIS_SPIRV_VAL_EXECUTABLE}"
            --target-env "${RACHIS_SHADER_SPIRV_ENVIRONMENT}"
            "${COOK_OUTPUT}"
        ${depfile_command}
        COMMAND
            "${CMAKE_COMMAND}"
            "-DOUTPUT=${key_path}"
            "-DSOURCE=${COOK_SOURCE}"
            "-DDEPFILE=${COOK_OUTPUT}.d"
            "-DCOMPILER=${RACHIS_SLANGC_EXECUTABLE}"
            "-DTARGET=spirv"
            "-DPROFILE=${COOK_PROFILE}"
            "-DENTRY=${COOK_ENTRY}"
            "-DENVIRONMENT=${RACHIS_SHADER_SPIRV_ENVIRONMENT}"
            "-DOPTIONS=${option_text}"
            "-DDEFINES=${define_text}"
            -P "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
        DEPENDS
            "${COOK_SOURCE}"
            "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
            "${RACHIS_SHADER_DEPFILE_SCRIPT}"
            ${COOK_DEPENDS}
        DEPFILE "${COOK_OUTPUT}.d"
        COMMENT "${COOK_COMMENT}"
        VERBATIM
    )
endfunction()

# Compiles a Slang entry point to DXIL with slangc, which validates and signs it through dxcompiler.
function(rachis_cook_slang_dxil)
    cmake_parse_arguments(COOK "" "SOURCE;ENTRY;STAGE;PROFILE;OUTPUT;COMMENT" "DEFINES;DEPENDS;INCLUDE_DIRECTORIES"
        ${ARGN})
    if(NOT COOK_SOURCE OR NOT COOK_ENTRY OR NOT COOK_OUTPUT)
        message(FATAL_ERROR "rachis_cook_slang_dxil requires SOURCE, ENTRY, and OUTPUT")
    endif()
    if(NOT RACHIS_SLANGC_EXECUTABLE)
        message(FATAL_ERROR "rachis_cook_slang_dxil requires RACHIS_SLANGC_EXECUTABLE")
    endif()
    if(NOT COOK_PROFILE)
        set(COOK_PROFILE "${RACHIS_SHADER_SLANG_DEFAULT_PROFILE}")
    endif()
    set(key_path "${COOK_OUTPUT}.key")
    set(define_args)
    foreach(define IN LISTS COOK_DEFINES)
        list(APPEND define_args "-D${define}")
    endforeach()
    list(JOIN COOK_DEFINES "," define_text)
    set(stage_args)
    if(COOK_STAGE)
        set(stage_args -stage "${COOK_STAGE}")
    endif()
    set(compile_options ${RACHIS_SHADER_SLANG_DXIL_OPTIONS} -profile "${COOK_PROFILE}" ${stage_args})
    foreach(directory IN LISTS COOK_INCLUDE_DIRECTORIES)
        list(APPEND compile_options -I "${directory}")
    endforeach()
    list(JOIN compile_options " " option_text)
    cmake_path(GET COOK_OUTPUT PARENT_PATH output_directory)
    rachis_shader_depfile_command(depfile_command slangc "${COOK_OUTPUT}")
    add_custom_command(
        OUTPUT "${COOK_OUTPUT}" "${key_path}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND
            "${RACHIS_SLANGC_EXECUTABLE}"
            "${COOK_SOURCE}"
            ${compile_options}
            ${define_args}
            -entry "${COOK_ENTRY}"
            -o "${COOK_OUTPUT}"
            -depfile "${COOK_OUTPUT}.slangc.d"
        ${depfile_command}
        COMMAND
            "${CMAKE_COMMAND}"
            "-DOUTPUT=${key_path}"
            "-DSOURCE=${COOK_SOURCE}"
            "-DDEPFILE=${COOK_OUTPUT}.d"
            "-DCOMPILER=${RACHIS_SLANGC_EXECUTABLE}"
            "-DCOMPILER_VERSION=${RACHIS_SLANGC_VERSION}"
            "-DTARGET=dxil"
            "-DPROFILE=${COOK_PROFILE}"
            "-DENTRY=${COOK_ENTRY}"
            "-DENVIRONMENT=d3d12"
            "-DOPTIONS=${option_text}"
            "-DDEFINES=${define_text}"
            -P "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
        DEPENDS
            "${COOK_SOURCE}"
            "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
            "${RACHIS_SHADER_DEPFILE_SCRIPT}"
            ${COOK_DEPENDS}
        DEPFILE "${COOK_OUTPUT}.d"
        COMMENT "${COOK_COMMENT}"
        VERBATIM
    )
endfunction()

# Compiles a GLSL shader to SPIR-V for Vulkan with glslang and validates it. GLSL's entry function is main; ENTRY
# names the SPIR-V entry point, which glslang renames main to.
function(rachis_cook_glslang_spirv)
    cmake_parse_arguments(COOK "" "SOURCE;ENTRY;STAGE;OUTPUT;COMMENT" "DEFINES;DEPENDS" ${ARGN})
    if(NOT COOK_SOURCE OR NOT COOK_ENTRY OR NOT COOK_STAGE OR NOT COOK_OUTPUT)
        message(FATAL_ERROR "rachis_cook_glslang_spirv requires SOURCE, ENTRY, STAGE, and OUTPUT")
    endif()
    if(NOT RACHIS_GLSLANG_EXECUTABLE)
        message(FATAL_ERROR "rachis_cook_glslang_spirv requires RACHIS_GLSLANG_EXECUTABLE")
    endif()
    if(COOK_STAGE STREQUAL "vertex")
        set(glslang_stage vert)
    elseif(COOK_STAGE STREQUAL "pixel")
        set(glslang_stage frag)
    elseif(COOK_STAGE STREQUAL "compute")
        set(glslang_stage comp)
    else()
        message(FATAL_ERROR "rachis_cook_glslang_spirv: STAGE must be vertex, pixel, or compute.")
    endif()
    set(key_path "${COOK_OUTPUT}.key")
    set(define_args)
    foreach(define IN LISTS COOK_DEFINES)
        list(APPEND define_args "-D${define}")
    endforeach()
    list(JOIN COOK_DEFINES "," define_text)
    set(compile_options ${RACHIS_SHADER_GLSLANG_SPIRV_OPTIONS} -S ${glslang_stage})
    list(JOIN compile_options " " option_text)
    cmake_path(GET COOK_OUTPUT PARENT_PATH output_directory)
    rachis_shader_depfile_command(depfile_command glslang "${COOK_OUTPUT}")
    add_custom_command(
        OUTPUT "${COOK_OUTPUT}" "${key_path}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND
            "${RACHIS_GLSLANG_EXECUTABLE}"
            ${compile_options}
            -e "${COOK_ENTRY}"
            --source-entrypoint main
            ${define_args}
            --depfile "${COOK_OUTPUT}.glslang.d"
            -o "${COOK_OUTPUT}"
            "${COOK_SOURCE}"
        COMMAND
            "${RACHIS_SPIRV_VAL_EXECUTABLE}"
            --target-env "${RACHIS_SHADER_SPIRV_ENVIRONMENT}"
            "${COOK_OUTPUT}"
        ${depfile_command}
        COMMAND
            "${CMAKE_COMMAND}"
            "-DOUTPUT=${key_path}"
            "-DSOURCE=${COOK_SOURCE}"
            "-DDEPFILE=${COOK_OUTPUT}.d"
            "-DCOMPILER=${RACHIS_GLSLANG_EXECUTABLE}"
            "-DTARGET=spirv"
            "-DPROFILE=${glslang_stage}"
            "-DENTRY=${COOK_ENTRY}"
            "-DENVIRONMENT=${RACHIS_SHADER_SPIRV_ENVIRONMENT}"
            "-DOPTIONS=${option_text}"
            "-DDEFINES=${define_text}"
            -P "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
        DEPENDS
            "${COOK_SOURCE}"
            "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
            "${RACHIS_SHADER_DEPFILE_SCRIPT}"
            ${COOK_DEPENDS}
        DEPFILE "${COOK_OUTPUT}.d"
        COMMENT "${COOK_COMMENT}"
        VERBATIM
    )
endfunction()

function(rachis_cook_slang_metallib)
    rachis_ensure_shader_binary_tool(rachis_shader_binary_tool_cmd rachis_shader_binary_tool_depends)
    cmake_parse_arguments(COOK "NO_WARNINGS_AS_ERRORS" "SOURCE;ENTRY;STAGE;PROFILE;OUTPUT;COMMENT"
        "DEFINES;DEPENDS;INCLUDE_DIRECTORIES" ${ARGN})
    if(NOT COOK_SOURCE OR NOT COOK_ENTRY OR NOT COOK_OUTPUT)
        message(FATAL_ERROR "rachis_cook_slang_metallib requires SOURCE, ENTRY, and OUTPUT")
    endif()
    if(NOT RACHIS_SLANGC_EXECUTABLE)
        message(FATAL_ERROR "rachis_cook_slang_metallib requires RACHIS_SLANGC_EXECUTABLE")
    endif()
    if(NOT COOK_PROFILE)
        set(COOK_PROFILE "${RACHIS_SHADER_SLANG_DEFAULT_PROFILE}")
    endif()
    list(FIND COOK_DEFINES "RACHIS_METAL=1" rachis_metal_define_index)
    if(rachis_metal_define_index EQUAL -1)
        list(PREPEND COOK_DEFINES RACHIS_METAL=1)
    endif()
    set(key_path "${COOK_OUTPUT}.key")
    set(json_path "${COOK_OUTPUT}.json")
    set(metal_path "${COOK_OUTPUT}.metal")
    string(REGEX REPLACE "\\.metallib$" ".pnms" pnms_path "${COOK_OUTPUT}")
    set(define_args)
    foreach(define IN LISTS COOK_DEFINES)
        list(APPEND define_args --define "${define}")
    endforeach()
    list(JOIN COOK_DEFINES "," define_text)
    set(warning_args)
    set(cook_extra_args --profile "${COOK_PROFILE}")
    if(COOK_NO_WARNINGS_AS_ERRORS)
        list(APPEND cook_extra_args --no-warnings-as-errors)
    else()
        list(APPEND warning_args -warnings-as-errors all)
    endif()
    set(metallib_options
        cook-metallib-v4 ${RACHIS_SHADER_SLANG_METALLIB_OPTIONS} -profile "${COOK_PROFILE}" ${warning_args})
    if(COOK_STAGE)
        list(APPEND cook_extra_args --stage "${COOK_STAGE}")
        list(APPEND metallib_options -stage "${COOK_STAGE}")
    endif()
    foreach(directory IN LISTS COOK_INCLUDE_DIRECTORIES)
        list(APPEND cook_extra_args --include-dir "${directory}")
        list(APPEND metallib_options -I "${directory}")
    endforeach()
    list(JOIN metallib_options " " option_text)
    set(cook_outputs "${COOK_OUTPUT}" "${key_path}" "${json_path}" "${pnms_path}")
    if(RACHIS_METAL_EMIT_DEBUG_SOURCE)
        list(APPEND cook_outputs "${metal_path}")
        list(APPEND cook_extra_args --metal-source "${metal_path}")
    endif()
    cmake_path(GET COOK_OUTPUT PARENT_PATH output_directory)
    rachis_shader_depfile_command(depfile_command slangc "${COOK_OUTPUT}")
    add_custom_command(
        OUTPUT ${cook_outputs}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND
            "${rachis_shader_binary_tool_cmd}"
            cook-metallib
            --slangc "${RACHIS_SLANGC_EXECUTABLE}"
            --source "${COOK_SOURCE}"
            --entry "${COOK_ENTRY}"
            --output "${COOK_OUTPUT}"
            --json "${json_path}"
            --pnms "${pnms_path}"
            --depfile "${COOK_OUTPUT}.slangc.d"
            ${cook_extra_args}
            ${define_args}
        ${depfile_command}
        COMMAND
            "${CMAKE_COMMAND}"
            "-DOUTPUT=${key_path}"
            "-DSOURCE=${COOK_SOURCE}"
            "-DDEPFILE=${COOK_OUTPUT}.d"
            "-DCOMPILER=${RACHIS_SLANGC_EXECUTABLE}"
            "-DCOMPILER_VERSION=${RACHIS_SLANGC_VERSION}"
            "-DTARGET=metallib"
            "-DPROFILE=${COOK_PROFILE}"
            "-DENTRY=${COOK_ENTRY}"
            "-DENVIRONMENT=${RACHIS_SHADER_METAL_ENVIRONMENT}"
            "-DSDK=${RACHIS_APPLE_SDK_VERSION}"
            "-DMSL=${RACHIS_SHADER_METAL_MSL}"
            "-DOPTIONS=${option_text}"
            "-DDEFINES=${define_text}"
            -P "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
        DEPENDS
            "${COOK_SOURCE}"
            "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
            "${RACHIS_SHADER_DEPFILE_SCRIPT}"
            ${rachis_shader_binary_tool_depends}
            ${COOK_DEPENDS}
        DEPFILE "${COOK_OUTPUT}.d"
        COMMENT "${COOK_COMMENT}"
        VERBATIM
    )
endfunction()

function(rachis_cook_dxc_spirv)
    cmake_parse_arguments(COOK "" "SOURCE;ENTRY;PROFILE;OUTPUT;HEADER;SYMBOL;COMMENT"
        "DEFINES;DEPENDS;OPTIONS;INCLUDE_DIRECTORIES" ${ARGN})
    if(NOT COOK_SOURCE OR NOT COOK_ENTRY OR NOT COOK_PROFILE OR NOT COOK_OUTPUT)
        message(FATAL_ERROR "rachis_cook_dxc_spirv requires SOURCE, ENTRY, PROFILE, and OUTPUT")
    endif()
    set(key_path "${COOK_OUTPUT}.key")
    set(define_args)
    foreach(define IN LISTS COOK_DEFINES)
        list(APPEND define_args "-D${define}")
    endforeach()
    list(JOIN COOK_DEFINES "," define_text)
    set(header_args)
    if(COOK_HEADER)
        if(NOT COOK_SYMBOL)
            message(FATAL_ERROR "rachis_cook_dxc_spirv HEADER requires SYMBOL")
        endif()
        list(APPEND header_args -Fh "${COOK_HEADER}" -Vn "${COOK_SYMBOL}")
    endif()
    set(compile_options ${RACHIS_SHADER_DXC_SPIRV_OPTIONS} ${COOK_OPTIONS})
    foreach(directory IN LISTS COOK_INCLUDE_DIRECTORIES)
        list(APPEND compile_options -I "${directory}")
    endforeach()
    list(JOIN compile_options " " option_text)
    cmake_path(GET COOK_OUTPUT PARENT_PATH output_directory)
    rachis_shader_depfile_command(depfile_command dxc "${COOK_OUTPUT}")
    add_custom_command(
        OUTPUT "${COOK_OUTPUT}" "${key_path}" ${COOK_HEADER}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
        COMMAND
            "${RACHIS_DXC_SPIRV_EXECUTABLE}"
            -T "${COOK_PROFILE}"
            -E "${COOK_ENTRY}"
            ${compile_options}
            ${define_args}
            -Fo "${COOK_OUTPUT}"
            ${header_args}
            "${COOK_SOURCE}"
        COMMAND
            "${RACHIS_DXC_SPIRV_EXECUTABLE}"
            -T "${COOK_PROFILE}"
            -E "${COOK_ENTRY}"
            ${compile_options}
            ${define_args}
            -MF "${COOK_OUTPUT}.dxc.d"
            "${COOK_SOURCE}"
        COMMAND
            "${RACHIS_SPIRV_VAL_EXECUTABLE}"
            --target-env "${RACHIS_SHADER_SPIRV_ENVIRONMENT}"
            "${COOK_OUTPUT}"
        ${depfile_command}
        COMMAND
            "${CMAKE_COMMAND}"
            "-DOUTPUT=${key_path}"
            "-DSOURCE=${COOK_SOURCE}"
            "-DDEPFILE=${COOK_OUTPUT}.d"
            "-DCOMPILER=${RACHIS_DXC_SPIRV_EXECUTABLE}"
            "-DTARGET=spirv"
            "-DPROFILE=${COOK_PROFILE}"
            "-DENTRY=${COOK_ENTRY}"
            "-DENVIRONMENT=${RACHIS_SHADER_SPIRV_ENVIRONMENT}"
            "-DOPTIONS=${option_text}"
            "-DDEFINES=${define_text}"
            -P "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
        DEPENDS
            "${COOK_SOURCE}"
            "${RACHIS_SHADER_COOKING_KEY_SCRIPT}"
            "${RACHIS_SHADER_DEPFILE_SCRIPT}"
            ${COOK_DEPENDS}
        DEPFILE "${COOK_OUTPUT}.d"
        COMMENT "${COOK_COMMENT}"
        VERBATIM
    )
endfunction()
