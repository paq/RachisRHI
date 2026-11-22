cmake_minimum_required(VERSION 4.0)
project(RachisShaderLanguage LANGUAGES CXX)

find_package(Rachis 0.5 CONFIG REQUIRED)

add_library(RachisShaderLanguage STATIC "${PROJECT_SOURCE_DIR}/../First.cpp")
target_compile_features(RachisShaderLanguage PRIVATE cxx_std_23)
target_link_libraries(RachisShaderLanguage PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
# RACHIS_SHADER_LANGUAGE_CASE picks the mistake: a source whose extension names no language, or a LANGUAGE that is
# not one rachis_add_shader compiles.
if(RACHIS_SHADER_LANGUAGE_CASE STREQUAL "extension")
    rachis_add_shader(RachisShaderLanguage
        NAME FillShader
        SOURCE "${PROJECT_SOURCE_DIR}/../Fill.shader"
        ENTRY fillMain
        STAGE compute)
elseif(RACHIS_SHADER_LANGUAGE_CASE STREQUAL "language")
    rachis_add_shader(RachisShaderLanguage
        NAME FillShader
        SOURCE "${PROJECT_SOURCE_DIR}/../Fill.hlsl"
        LANGUAGE MSL
        ENTRY fillMain
        STAGE compute)
else()
    message(FATAL_ERROR "Set RACHIS_SHADER_LANGUAGE_CASE to extension or language.")
endif()
