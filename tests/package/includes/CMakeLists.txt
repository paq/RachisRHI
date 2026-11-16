cmake_minimum_required(VERSION 4.0)
project(RachisShaderIncludes LANGUAGES CXX)

find_package(Rachis 0.2 CONFIG REQUIRED)
enable_testing()

set(rachis_shader_includes_backends)
foreach(backend IN ITEMS D3D12 VULKAN METAL)
    get_target_property(rachis_enable_${backend} Rachis::RHI RACHIS_ENABLE_${backend})
    if(rachis_enable_${backend} MATCHES "-NOTFOUND$")
        message(FATAL_ERROR "Rachis::RHI does not export RACHIS_ENABLE_${backend}.")
    endif()
    if(rachis_enable_${backend})
        string(TOLOWER "${backend}" backend)
        list(APPEND rachis_shader_includes_backends "${backend}")
    endif()
endforeach()
list(JOIN rachis_shader_includes_backends "|" rachis_shader_includes_backends)

add_executable(RachisShaderIncludes "${PROJECT_SOURCE_DIR}/main.cpp")
target_compile_features(RachisShaderIncludes PRIVATE cxx_std_23)
target_link_libraries(RachisShaderIncludes PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
rachis_add_shader(RachisShaderIncludes
    NAME FillShader
    SOURCE "${PROJECT_SOURCE_DIR}/shaders/Fill.hlsl"
    ENTRY fillMain
    STAGE compute)

add_test(NAME RachisShaderIncludes
    COMMAND "${CMAKE_COMMAND}"
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DCONFIG=$<CONFIG>"
        "-DTARGET=RachisShaderIncludes"
        "-DSHADERS=${CMAKE_CURRENT_BINARY_DIR}/rachis_shaders/RachisShaderIncludes"
        "-DINCLUDE=${PROJECT_SOURCE_DIR}/shaders/fill values/FillValue.hlsli"
        "-DBACKENDS=${rachis_shader_includes_backends}"
        -P "${PROJECT_SOURCE_DIR}/CheckIncludes.cmake")
