enable_language(C)

unset(RACHIS_VULKAN_HEADERS_TAG CACHE)
unset(RACHIS_VOLK_TAG CACHE)
unset(RACHIS_VMA_TAG CACHE)
set(RACHIS_VULKAN_HEADERS_TAG "v1.3.296")
set(RACHIS_VOLK_TAG "vulkan-sdk-1.3.296.0")
set(RACHIS_VMA_TAG "v3.2.1")
set(rachis_vulkan_header_version_minimum 296)
set(rachis_vulkan_headers_commit 29f979ee5aa58b7b005f805ea8df7a855c39ff37)
set(rachis_volk_commit 59d26900f53c7621a8ba8ab0e3f18d3bd883fa9a)
set(rachis_vma_commit c788c52156f3ef7bc7ab769cb03c110a53ac8fcb)

if(TARGET Vulkan-Headers)
    get_target_property(rachis_vulkan_headers_includes Vulkan-Headers INTERFACE_INCLUDE_DIRECTORIES)
    string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]*)>" "\\1"
        rachis_vulkan_headers_includes "${rachis_vulkan_headers_includes}")
    string(REGEX REPLACE "\\$<INSTALL_INTERFACE:[^>]*>" ""
        rachis_vulkan_headers_includes "${rachis_vulkan_headers_includes}")
    set(rachis_vulkan_core_header "")
    foreach(rachis_vulkan_headers_include IN LISTS rachis_vulkan_headers_includes)
        if(EXISTS "${rachis_vulkan_headers_include}/vulkan/vulkan_core.h")
            set(rachis_vulkan_core_header "${rachis_vulkan_headers_include}/vulkan/vulkan_core.h")
            break()
        endif()
    endforeach()
    if(NOT rachis_vulkan_core_header)
        message(FATAL_ERROR
            "RachisRHI reuses the existing Vulkan-Headers target, but none of its include directories "
            "(${rachis_vulkan_headers_includes}) contains vulkan/vulkan_core.h.")
    endif()
    file(STRINGS "${rachis_vulkan_core_header}" rachis_vulkan_header_version
        REGEX "^#define[ \t]+VK_HEADER_VERSION[ \t]+[0-9]+")
    string(REGEX MATCH "[0-9]+$" rachis_vulkan_header_version "${rachis_vulkan_header_version}")
    if(NOT rachis_vulkan_header_version OR rachis_vulkan_header_version LESS rachis_vulkan_header_version_minimum)
        message(FATAL_ERROR
            "RachisRHI needs Vulkan-Headers with VK_HEADER_VERSION ${rachis_vulkan_header_version_minimum} or newer, "
            "but the existing Vulkan-Headers target provides '${rachis_vulkan_header_version}' "
            "(${rachis_vulkan_core_header}).")
    endif()
    message(STATUS
        "RachisRHI: using the existing Vulkan-Headers target (VK_HEADER_VERSION ${rachis_vulkan_header_version})")
else()
    FetchContent_Declare(
        RachisVulkanHeaders
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG ${rachis_vulkan_headers_commit}
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(RachisVulkanHeaders)
    list(APPEND rachis_fetched_dependencies RachisVulkanHeaders)
    if(NOT TARGET Vulkan-Headers)
        message(FATAL_ERROR
            "Vulkan-Headers ${RACHIS_VULKAN_HEADERS_TAG} did not define the Vulkan-Headers target. "
            "Check the FetchContent clone under the CMake build directory.")
    endif()
endif()

set(rachis_volk_defines VK_ENABLE_BETA_EXTENSIONS)
if(WIN32)
    list(APPEND rachis_volk_defines VK_USE_PLATFORM_WIN32_KHR)
endif()

if(TARGET volk)
    get_target_property(rachis_volk_type volk TYPE)
    if(NOT rachis_volk_type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR
            "RachisRHI reuses the existing volk target, which must be a static library (got ${rachis_volk_type}).")
    endif()
    get_target_property(rachis_volk_interface_defines volk INTERFACE_COMPILE_DEFINITIONS)
    foreach(rachis_volk_define IN LISTS rachis_volk_defines)
        if(NOT rachis_volk_define IN_LIST rachis_volk_interface_defines)
            message(FATAL_ERROR
                "RachisRHI reuses the existing volk target, which must be built with ${rachis_volk_define}. "
                "Configure volk with VOLK_STATIC_DEFINES=\"${rachis_volk_defines}\".")
        endif()
    endforeach()
    get_target_property(rachis_volk_imported volk IMPORTED)
    if(NOT rachis_volk_imported)
        set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    message(STATUS "RachisRHI: using the existing volk target")
else()
    set(VOLK_PULL_IN_VULKAN OFF)
    set(VOLK_INSTALL OFF)
    set(VOLK_HEADERS_ONLY OFF)
    set(VOLK_STATIC_DEFINES "${rachis_volk_defines}")
    FetchContent_Declare(
        RachisVolk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG ${rachis_volk_commit}
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(RachisVolk)
    list(APPEND rachis_fetched_dependencies RachisVolk)
    if(NOT TARGET volk)
        message(FATAL_ERROR
            "volk ${RACHIS_VOLK_TAG} did not export the volk target. "
            "Install Git and allow FetchContent to clone https://github.com/zeux/volk.git.")
    endif()
    target_link_libraries(volk PUBLIC Vulkan-Headers)
    set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON DEBUG_POSTFIX d)
endif()

if(TARGET VulkanMemoryAllocator)
    message(STATUS "RachisRHI: using the existing VulkanMemoryAllocator target")
else()
    set(VMA_BUILD_DOCUMENTATION OFF)
    set(VMA_BUILD_SAMPLES OFF)
    set(VMA_ENABLE_INSTALL OFF)
    FetchContent_Declare(
        RachisVma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG ${rachis_vma_commit}
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(RachisVma)
    list(APPEND rachis_fetched_dependencies RachisVma)
    if(NOT TARGET VulkanMemoryAllocator)
        add_library(VulkanMemoryAllocator INTERFACE)
        target_include_directories(VulkanMemoryAllocator INTERFACE "${rachisvma_SOURCE_DIR}/include")
        add_library(GPUOpen::VulkanMemoryAllocator ALIAS VulkanMemoryAllocator)
    endif()
    target_compile_definitions(VulkanMemoryAllocator
        INTERFACE
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    )
endif()
