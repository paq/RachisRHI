cmake_minimum_required(VERSION 4.0)
project(RachisShaderReflectWithoutSlangc LANGUAGES CXX)

# Hides every slangc that rachis_add_shader could find, as tests/package/subdirectories does.
file(GLOB rachis_package_vulkan_sdk_bins "C:/VulkanSDK/*/Bin")
set(rachis_package_slangc_roots
    "$ENV{VULKAN_SDK}/Bin"
    "$ENV{VULKAN_SDK}/bin"
    "$ENV{VULKAN_SDK}/macOS/bin"
    ${rachis_package_vulkan_sdk_bins}
    /usr/bin
    /usr/local/bin
    /opt/homebrew/bin)
cmake_path(CONVERT "$ENV{PATH}" TO_CMAKE_PATH_LIST rachis_package_path)
foreach(directory IN LISTS rachis_package_slangc_roots rachis_package_path)
    if(EXISTS "${directory}/slangc" OR EXISTS "${directory}/slangc.exe")
        list(APPEND CMAKE_IGNORE_PATH "${directory}")
    endif()
endforeach()
foreach(attempt RANGE 16)
    unset(rachis_package_slangc)
    find_program(rachis_package_slangc NAMES slangc PATHS ${rachis_package_slangc_roots} NO_CACHE)
    if(NOT rachis_package_slangc)
        break()
    endif()
    if(attempt EQUAL 16)
        message(FATAL_ERROR "The test could not hide slangc at ${rachis_package_slangc}.")
    endif()
    get_filename_component(rachis_package_slangc_dir "${rachis_package_slangc}" DIRECTORY)
    list(APPEND CMAKE_IGNORE_PATH "${rachis_package_slangc_dir}")
endforeach()

find_package(Rachis 0.1 CONFIG REQUIRED)

add_library(RachisShaderReflectWithoutSlangc STATIC "${PROJECT_SOURCE_DIR}/../First.cpp")
target_compile_features(RachisShaderReflectWithoutSlangc PRIVATE cxx_std_23)
target_link_libraries(RachisShaderReflectWithoutSlangc PRIVATE $<COMPILE_ONLY:Rachis::RHI>)
rachis_add_shader(RachisShaderReflectWithoutSlangc
    NAME FillShader
    SOURCE "${PROJECT_SOURCE_DIR}/../Fill.hlsl"
    ENTRY fillMain
    STAGE compute
    DEFINES FILL_OFFSET=1
    REFLECT)
