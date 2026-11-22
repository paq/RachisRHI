cmake_minimum_required(VERSION 4.0)
project(RachisShaderNamesReuse LANGUAGES CXX)

find_package(Rachis 0.5 CONFIG REQUIRED)

add_library(RachisShaderNamesReuse STATIC "${PROJECT_SOURCE_DIR}/../First.cpp")
target_compile_features(RachisShaderNamesReuse PRIVATE cxx_std_23)
target_link_libraries(RachisShaderNamesReuse PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
foreach(offset IN ITEMS 1 2)
    rachis_add_shader(RachisShaderNamesReuse
        NAME FillShader
        SOURCE "${PROJECT_SOURCE_DIR}/../Fill.hlsl"
        ENTRY fillMain
        STAGE compute
        DEFINES FILL_OFFSET=${offset})
endforeach()
