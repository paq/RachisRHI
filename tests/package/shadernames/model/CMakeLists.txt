cmake_minimum_required(VERSION 4.0)
project(RachisShaderModel LANGUAGES CXX)

find_package(Rachis 0.2 CONFIG REQUIRED)

add_library(RachisShaderModel STATIC "${PROJECT_SOURCE_DIR}/../First.cpp")
target_compile_features(RachisShaderModel PRIVATE cxx_std_23)
target_link_libraries(RachisShaderModel PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
# SHADER_MODEL takes only the shader models from 6_0 to 6_6.
rachis_add_shader(RachisShaderModel
    NAME FillShader
    SOURCE "${PROJECT_SOURCE_DIR}/../Fill.hlsl"
    ENTRY fillMain
    STAGE compute
    SHADER_MODEL 6_7)
