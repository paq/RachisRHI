cmake_minimum_required(VERSION 4.0)
project(RachisShaderBuildPaths LANGUAGES CXX)

find_package(Rachis 0.5 CONFIG REQUIRED)
enable_testing()

foreach(backend IN ITEMS D3D12 VULKAN METAL)
    get_target_property(rachis_enable_${backend} Rachis::RHI RACHIS_ENABLE_${backend})
    if(rachis_enable_${backend} MATCHES "-NOTFOUND$")
        message(FATAL_ERROR "Rachis::RHI does not export RACHIS_ENABLE_${backend}.")
    endif()
endforeach()

add_executable(RachisShaderBuildPaths "${PROJECT_SOURCE_DIR}/main.cpp")
target_compile_features(RachisShaderBuildPaths PRIVATE cxx_std_23)
target_link_libraries(RachisShaderBuildPaths PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
rachis_add_shader(RachisShaderBuildPaths
    NAME Fill
    SOURCE "${PROJECT_SOURCE_DIR}/Fill.hlsl"
    ENTRY fillMain
    STAGE compute)

set(shaders "${CMAKE_CURRENT_BINARY_DIR}/rachis_shaders/RachisShaderBuildPaths")
set(binaries)
if(rachis_enable_D3D12)
    list(APPEND binaries "${shaders}/Fill.dxil")
endif()
if(rachis_enable_VULKAN)
    list(APPEND binaries "${shaders}/Fill.spv")
endif()
if(rachis_enable_METAL)
    list(APPEND binaries "${shaders}/Fill.pnms")
endif()
add_test(NAME RachisShaderBuildPaths COMMAND "${CMAKE_COMMAND}" -E sha256sum ${binaries})
