cmake_minimum_required(VERSION 4.0)
project(RachisShaderSubdirectories LANGUAGES C CXX)

option(RACHIS_PACKAGE_WITHOUT_SLANGC "Hide every slangc that find_program could find" OFF)
if(RACHIS_PACKAGE_WITHOUT_SLANGC)
    if(DEFINED ENV{SLANGC} OR DEFINED RACHIS_SLANGC_EXECUTABLE)
        message(FATAL_ERROR "RACHIS_PACKAGE_WITHOUT_SLANGC needs SLANGC and RACHIS_SLANGC_EXECUTABLE unset.")
    endif()
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
endif()

function(add_fill_library name)
    add_library(${name} STATIC "${PROJECT_SOURCE_DIR}/Fill.cpp")
    target_compile_features(${name} PRIVATE cxx_std_23)
    target_link_libraries(${name} PRIVATE Rachis::RHI)
    rachis_add_shader(${name}
        NAME FillShader
        SOURCE "${PROJECT_SOURCE_DIR}/Fill.hlsl"
        ENTRY fillMain
        STAGE compute)
endfunction()

if(RACHIS_SOURCE_DIR)
    include(CTest)
    add_subdirectory("${RACHIS_SOURCE_DIR}" rachis)
    add_fill_library(RachisShaderParent)
endif()
add_subdirectory(first)
add_subdirectory(second)
