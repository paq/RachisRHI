set(RACHIS_SHADER_BINARY_TOOL "" CACHE FILEPATH
    "Optional prebuilt host RachisShaderBinaryTool executable for cross-compilation")

function(rachis_define_shader_binary_tool)
    if(RACHIS_SHADER_BINARY_TOOL)
        if(NOT EXISTS "${RACHIS_SHADER_BINARY_TOOL}")
            message(FATAL_ERROR
                "RACHIS_SHADER_BINARY_TOOL does not exist: '${RACHIS_SHADER_BINARY_TOOL}'.")
        endif()
        return()
    endif()
    if(TARGET RachisShaderBinaryTool OR TARGET Rachis::ShaderBinaryTool)
        return()
    endif()

    get_filename_component(rhi_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    if(NOT EXISTS "${rhi_root}/cmake/RachisShaderBinaryTool.cpp")
        message(FATAL_ERROR
            "RachisShaderBinaryTool is not available. Install the Rachis package with its "
            "shader tool, or set RACHIS_SHADER_BINARY_TOOL to a host build of it.")
    endif()
    add_executable(RachisShaderBinaryTool "${rhi_root}/cmake/RachisShaderBinaryTool.cpp")
    target_include_directories(RachisShaderBinaryTool PRIVATE "${rhi_root}/src")
    target_compile_features(RachisShaderBinaryTool PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(RachisShaderBinaryTool PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(RachisShaderBinaryTool PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    set_target_properties(RachisShaderBinaryTool PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shader-tools"
    )
endfunction()

# Returns the command that runs RachisShaderBinaryTool and what a custom command must depend on.
# In order of preference: RACHIS_SHADER_BINARY_TOOL, the tool built by this source tree, the tool
# installed with the Rachis package (Rachis::ShaderBinaryTool), or a new build from source.
function(rachis_ensure_shader_binary_tool out_command out_depends)
    if(NOT RACHIS_SHADER_BINARY_TOOL AND CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR
            "RachisShaderBinaryTool runs during the build, and a build of it for the target cannot run "
            "on the host. Set RACHIS_SHADER_BINARY_TOOL to a host build of it when cross-compiling.")
    endif()
    rachis_define_shader_binary_tool()
    if(RACHIS_SHADER_BINARY_TOOL)
        set("${out_command}" "${RACHIS_SHADER_BINARY_TOOL}" PARENT_SCOPE)
        set("${out_depends}" "${RACHIS_SHADER_BINARY_TOOL}" PARENT_SCOPE)
    elseif(TARGET RachisShaderBinaryTool)
        set("${out_command}" "$<TARGET_FILE:RachisShaderBinaryTool>" PARENT_SCOPE)
        set("${out_depends}" RachisShaderBinaryTool PARENT_SCOPE)
    else()
        set("${out_command}" "$<TARGET_FILE:Rachis::ShaderBinaryTool>" PARENT_SCOPE)
        set("${out_depends}" "$<TARGET_FILE:Rachis::ShaderBinaryTool>" PARENT_SCOPE)
    endif()
endfunction()
