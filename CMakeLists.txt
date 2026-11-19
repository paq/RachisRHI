cmake_minimum_required(VERSION 4.0)
set(RACHIS_PACKAGE_VERSION 0.2.0)
project(RachisRHI VERSION ${RACHIS_PACKAGE_VERSION} LANGUAGES C CXX)
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
endif()

option(RACHIS_BUILD_TESTS
    "Build the RachisRHI tests and compile the RACHIS_TESTING hooks into the libraries when BUILD_TESTING is also on"
    ${PROJECT_IS_TOP_LEVEL})
option(RACHIS_INSTALL "Generate the RachisRHI install rules and CMake package" ${PROJECT_IS_TOP_LEVEL})
if(RACHIS_BUILD_TESTS AND BUILD_TESTING)
    set(rachis_build_tests ON)
else()
    set(rachis_build_tests OFF)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_C_COMPILER_LOADED)
    enable_language(C)
endif()

if(WIN32)
elseif(APPLE)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        message(FATAL_ERROR
            "macOS native builds support Apple silicon (arm64) only "
            "(got CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}'). Intel macOS is rejected.")
    endif()
    if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        message(FATAL_ERROR
            "macOS native builds support arm64 only "
            "(got CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}').")
    endif()
elseif(UNIX)
else()
    message(FATAL_ERROR "RachisRHI supports Windows, Linux, and Apple silicon macOS hosts.")
endif()

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(WIN32)
    set(RACHIS_BACKENDS "both" CACHE STRING "Windows RHI backends: d3d12, vulkan, or both")
    set_property(CACHE RACHIS_BACKENDS PROPERTY STRINGS d3d12 vulkan both)
elseif(APPLE)
    set(RACHIS_BACKENDS "metal" CACHE STRING
        "macOS RHI backend: metal (production) or vulkan (MoltenVK investigation)")
    set_property(CACHE RACHIS_BACKENDS PROPERTY STRINGS metal vulkan)
else()
    set(RACHIS_BACKENDS "vulkan" CACHE STRING "Linux RHI backend: vulkan")
    set_property(CACHE RACHIS_BACKENDS PROPERTY STRINGS vulkan)
endif()

if(RACHIS_BACKENDS STREQUAL "d3d12")
    if(NOT WIN32)
        message(FATAL_ERROR "RACHIS_BACKENDS=d3d12 is Windows-only.")
    endif()
    set(RACHIS_ENABLE_D3D12 ON)
    set(RACHIS_ENABLE_VULKAN OFF)
elseif(RACHIS_BACKENDS STREQUAL "vulkan")
    set(RACHIS_ENABLE_D3D12 OFF)
    set(RACHIS_ENABLE_VULKAN ON)
elseif(RACHIS_BACKENDS STREQUAL "both")
    if(NOT WIN32)
        message(FATAL_ERROR
            "RACHIS_BACKENDS=both is Windows-only. Use vulkan on Linux; use metal or vulkan on macOS.")
    endif()
    set(RACHIS_ENABLE_D3D12 ON)
    set(RACHIS_ENABLE_VULKAN ON)
elseif(RACHIS_BACKENDS STREQUAL "metal")
    if(NOT APPLE)
        message(FATAL_ERROR "RACHIS_BACKENDS=metal is macOS Apple silicon only.")
    endif()
    set(RACHIS_ENABLE_D3D12 OFF)
    set(RACHIS_ENABLE_VULKAN OFF)
    set(RACHIS_ENABLE_METAL ON)
else()
    message(FATAL_ERROR
        "RACHIS_BACKENDS must be d3d12, vulkan, both, or metal (got '${RACHIS_BACKENDS}').")
endif()
if(NOT DEFINED RACHIS_ENABLE_METAL)
    set(RACHIS_ENABLE_METAL OFF)
endif()

option(RACHIS_ENABLE_DEBUG_MARKERS "Enable portable RHI debug markers through PIX on D3D12" ON)
if(NOT RACHIS_ENABLE_D3D12)
    set(RACHIS_ENABLE_DEBUG_MARKERS OFF)
endif()

if(RACHIS_ENABLE_METAL AND RACHIS_ENABLE_VULKAN)
    message(FATAL_ERROR
        "Metal and Vulkan cannot be enabled in the same native build. "
        "Use macos-arm64-metal for production Metal or macos-arm64-vulkan for the MoltenVK investigation.")
endif()
if(RACHIS_ENABLE_METAL AND RACHIS_ENABLE_D3D12)
    message(FATAL_ERROR "RACHIS_BACKENDS=metal cannot be combined with D3D12.")
endif()

if(RACHIS_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "RACHIS_ENABLE_METAL is macOS Apple silicon only.")
    endif()
    set(CMAKE_OBJCXX_STANDARD 23)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
    enable_language(OBJCXX)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET AND PROJECT_IS_TOP_LEVEL)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Metal floor is macOS 26.0" FORCE)
    elseif(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
    endif()
    if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "26.0")
        message(FATAL_ERROR
            "Metal requires CMAKE_OSX_DEPLOYMENT_TARGET 26.0 or newer "
            "(got '${CMAKE_OSX_DEPLOYMENT_TARGET}').")
    endif()
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-version
        OUTPUT_VARIABLE RACHIS_APPLE_SDK_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RACHIS_APPLE_SDK_VERSION_RESULT
    )
    if(NOT RACHIS_APPLE_SDK_VERSION_RESULT EQUAL 0 OR NOT RACHIS_APPLE_SDK_VERSION)
        message(FATAL_ERROR "xcrun --sdk macosx --show-sdk-version failed. Install Xcode 26+ command-line tools.")
    endif()
    if(RACHIS_APPLE_SDK_VERSION VERSION_LESS "26.0")
        message(FATAL_ERROR
            "Metal requires MacOSX SDK 26.0 or newer (got '${RACHIS_APPLE_SDK_VERSION}').")
    endif()
    find_library(RACHIS_METAL_FRAMEWORK Metal REQUIRED)
    find_library(RACHIS_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(RACHIS_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    find_library(RACHIS_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
    find_library(RACHIS_APPKIT_FRAMEWORK AppKit REQUIRED)
    message(STATUS "Rachis Metal: AppleClang ${CMAKE_CXX_COMPILER_VERSION}, SDK ${RACHIS_APPLE_SDK_VERSION}, "
        "deployment ${CMAKE_OSX_DEPLOYMENT_TARGET}, language Objective-C++ ARC. "
        "slangc 2026.17.1+ is required to cook metallib. "
        "Vulkan SDK, dxc, and spirv-val are not configure requirements.")
endif()

set(rachis_fetched_dependencies)
if(RACHIS_ENABLE_VULKAN)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RachisVulkanDependencies.cmake")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RachisShaderCooking.cmake")
set(rachis_shader_compiler_targets)
if(RACHIS_ENABLE_D3D12)
    list(APPEND rachis_shader_compiler_targets dxil)
endif()
if(RACHIS_ENABLE_VULKAN)
    list(APPEND rachis_shader_compiler_targets spirv)
    if(rachis_build_tests)
        list(APPEND rachis_shader_compiler_targets slang-spirv)
    endif()
endif()
if(RACHIS_ENABLE_METAL)
    list(APPEND rachis_shader_compiler_targets metallib)
endif()
rachis_find_shader_compilers(${rachis_shader_compiler_targets})
rachis_define_shader_binary_tool()
if(TARGET RachisShaderBinaryTool)
    add_executable(Rachis::ShaderBinaryTool ALIAS RachisShaderBinaryTool)
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RachisShaders.cmake")

if(RACHIS_ENABLE_DEBUG_MARKERS)
    FetchContent_Declare(
        winpixeventruntime
        URL https://api.nuget.org/v3-flatcontainer/winpixeventruntime/1.0.240308001/winpixeventruntime.1.0.240308001.nupkg
        URL_HASH SHA256=726ACC93D6968E2146261A1E415521747D50AD69894C2B42B5D0D4C29FD66EC4
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(winpixeventruntime)
    list(APPEND rachis_fetched_dependencies winpixeventruntime)

    if(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        set(rachis_winpix_architecture arm64)
    else()
        set(rachis_winpix_architecture x64)
    endif()

    add_library(RachisWinPixEventRuntime SHARED IMPORTED GLOBAL)
    set_target_properties(RachisWinPixEventRuntime PROPERTIES
        IMPORTED_LOCATION
            "${winpixeventruntime_SOURCE_DIR}/bin/${rachis_winpix_architecture}/WinPixEventRuntime.dll"
        IMPORTED_IMPLIB
            "${winpixeventruntime_SOURCE_DIR}/bin/${rachis_winpix_architecture}/WinPixEventRuntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES
            "${winpixeventruntime_SOURCE_DIR}/Include/WinPixEventRuntime"
    )
endif()

if(RACHIS_ENABLE_D3D12)
    if(TARGET D3D12MemoryAllocator)
        message(STATUS "RachisRHI: using the existing D3D12MemoryAllocator target")
    else()
        FetchContent_Declare(
            D3D12MemoryAllocator
            GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator.git
            GIT_TAG 1d86c1130f61453634b1df85782e1fecfd59a525
            EXCLUDE_FROM_ALL
        )
        block(PROPAGATE d3d12memoryallocator_SOURCE_DIR)
            set(BUILD_SHARED_LIBS OFF)
            set(BUILD_DOCUMENTATION OFF)
            set(D3D12MA_BUILD_SAMPLE OFF)
            FetchContent_MakeAvailable(D3D12MemoryAllocator)
        endblock()
        list(APPEND rachis_fetched_dependencies D3D12MemoryAllocator)
    endif()
endif()

set(rachis_rhi_public_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(rachis_rhi_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(RACHIS_ENABLE_D3D12)
set(rachis_rhi_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/Generated")
set(rachis_generate_mipmaps_1d_header "${rachis_rhi_generated_dir}/RachisGenerateMipmaps1D.h")
set(rachis_generate_mipmaps_header "${rachis_rhi_generated_dir}/RachisGenerateMipmaps.h")
set(rachis_generate_mipmaps_3d_header "${rachis_rhi_generated_dir}/RachisGenerateMipmaps3D.h")
set(rachis_resolve_depth_vertex_header "${rachis_rhi_generated_dir}/RachisResolveDepthVertex.h")
set(rachis_resolve_depth_pixel_header "${rachis_rhi_generated_dir}/RachisResolveDepthPixel.h")
set(rachis_rhi_dxil_options)
if(rachis_build_tests)
    set(rachis_rhi_dxil_options -WX)
endif()
rachis_cook_dxc_dxil(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/GenerateMipmaps.hlsl"
    ENTRY main
    PROFILE cs_6_0
    OPTIONS ${rachis_rhi_dxil_options}
    OUTPUT "${rachis_rhi_generated_dir}/RachisGenerateMipmaps1D.dxil"
    HEADER "${rachis_generate_mipmaps_1d_header}"
    SYMBOL g_rachisGenerateMipmaps1DDxil
    DEFINES RACHIS_MIPMAP_DIMENSION=1
    COMMENT "Compiling the Rachis D3D12 1D mipmap generator"
)
rachis_cook_dxc_dxil(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/GenerateMipmaps.hlsl"
    ENTRY main
    PROFILE cs_6_0
    OPTIONS ${rachis_rhi_dxil_options}
    OUTPUT "${rachis_rhi_generated_dir}/RachisGenerateMipmaps.dxil"
    HEADER "${rachis_generate_mipmaps_header}"
    SYMBOL g_rachisGenerateMipmapsDxil
    DEFINES RACHIS_MIPMAP_DIMENSION=2
    COMMENT "Compiling the Rachis D3D12 mipmap generator"
)
rachis_cook_dxc_dxil(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/GenerateMipmaps.hlsl"
    ENTRY main
    PROFILE cs_6_0
    OPTIONS ${rachis_rhi_dxil_options}
    OUTPUT "${rachis_rhi_generated_dir}/RachisGenerateMipmaps3D.dxil"
    HEADER "${rachis_generate_mipmaps_3d_header}"
    SYMBOL g_rachisGenerateMipmaps3DDxil
    DEFINES RACHIS_MIPMAP_DIMENSION=3
    COMMENT "Compiling the Rachis D3D12 3D mipmap generator"
)
rachis_cook_dxc_dxil(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/ResolveDepth.hlsl"
    ENTRY vertexMain
    PROFILE vs_6_0
    OPTIONS ${rachis_rhi_dxil_options}
    OUTPUT "${rachis_rhi_generated_dir}/RachisResolveDepthVertex.dxil"
    HEADER "${rachis_resolve_depth_vertex_header}"
    SYMBOL g_rachisResolveDepthVertexDxil
    COMMENT "Compiling the Rachis D3D12 depth resolver vertex shader"
)
rachis_cook_dxc_dxil(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/ResolveDepth.hlsl"
    ENTRY pixelMain
    PROFILE ps_6_0
    OPTIONS ${rachis_rhi_dxil_options}
    OUTPUT "${rachis_rhi_generated_dir}/RachisResolveDepthPixel.dxil"
    HEADER "${rachis_resolve_depth_pixel_header}"
    SYMBOL g_rachisResolveDepthPixelDxil
    COMMENT "Compiling the Rachis D3D12 depth resolver pixel shader"
)
add_custom_target(
    RachisRHIShaders
    DEPENDS
        "${rachis_generate_mipmaps_1d_header}"
        "${rachis_generate_mipmaps_header}"
        "${rachis_generate_mipmaps_3d_header}"
        "${rachis_resolve_depth_vertex_header}"
        "${rachis_resolve_depth_pixel_header}"
)
endif()

if(RACHIS_ENABLE_VULKAN)
    set(rachis_rhi_spirv_dir "${CMAKE_CURRENT_BINARY_DIR}/Generated/spirv")
    set(rachis_mipmaps_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/GenerateMipmaps.hlsl")
    set(rachis_mipmaps_spirv_options -fvk-b-shift 0 0 -fvk-t-shift 1 0 -fvk-u-shift 2 0 -fvk-s-shift 3 0)
    rachis_cook_dxc_spirv(
        SOURCE "${rachis_mipmaps_hlsl}"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GenerateMipmaps1D.spv"
        HEADER "${rachis_rhi_spirv_dir}/RachisGenerateMipmaps1D.spirv.h"
        SYMBOL g_rachisGenerateMipmaps1DSpirv
        DEFINES RACHIS_MIPMAP_DIMENSION=1
        OPTIONS ${rachis_mipmaps_spirv_options}
        COMMENT "Cooking Vulkan SPIR-V GenerateMipmaps1D"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${rachis_mipmaps_hlsl}"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GenerateMipmaps2D.spv"
        HEADER "${rachis_rhi_spirv_dir}/RachisGenerateMipmaps2D.spirv.h"
        SYMBOL g_rachisGenerateMipmaps2DSpirv
        DEFINES RACHIS_MIPMAP_DIMENSION=2
        OPTIONS ${rachis_mipmaps_spirv_options}
        COMMENT "Cooking Vulkan SPIR-V GenerateMipmaps2D"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${rachis_mipmaps_hlsl}"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GenerateMipmaps3D.spv"
        HEADER "${rachis_rhi_spirv_dir}/RachisGenerateMipmaps3D.spirv.h"
        SYMBOL g_rachisGenerateMipmaps3DSpirv
        DEFINES RACHIS_MIPMAP_DIMENSION=3
        OPTIONS ${rachis_mipmaps_spirv_options}
        COMMENT "Cooking Vulkan SPIR-V GenerateMipmaps3D"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/Vulkan/Shaders/CountedDispatchArguments.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/CountedDispatchArguments.spv"
        HEADER "${rachis_rhi_spirv_dir}/RachisCountedDispatchArguments.spirv.h"
        SYMBOL g_rachisCountedDispatchArgumentsSpirv
        COMMENT "Cooking Vulkan SPIR-V CountedDispatchArguments"
    )
    add_custom_target(RachisRHISpirvShaders ALL
        DEPENDS
            "${rachis_rhi_spirv_dir}/GenerateMipmaps1D.spv"
            "${rachis_rhi_spirv_dir}/GenerateMipmaps2D.spv"
            "${rachis_rhi_spirv_dir}/GenerateMipmaps3D.spv"
            "${rachis_rhi_spirv_dir}/CountedDispatchArguments.spv"
    )
endif()

if(RACHIS_ENABLE_VULKAN AND rachis_build_tests)
    function(rachis_rhi_slang_spirv name source entry)
        cmake_parse_arguments(SPIRV "" "" "DEFINES" ${ARGN})
        rachis_cook_slang_spirv(
            SOURCE "${source}"
            ENTRY "${entry}"
            OUTPUT "${rachis_rhi_spirv_dir}/${name}"
            DEFINES ${SPIRV_DEFINES}
            COMMENT "Cooking Vulkan SPIR-V ${name}"
        )
    endfunction()
    rachis_rhi_slang_spirv(BootstrapTriangle.vs.spv "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapTriangle.slang" vertexMain)
    rachis_rhi_slang_spirv(BootstrapTriangle.ps.spv "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapTriangle.slang" pixelMain)
    rachis_rhi_slang_spirv(BootstrapCube.vs.spv "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapCube.slang" vertexMain)
    rachis_rhi_slang_spirv(BootstrapCube.ps.spv "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapCube.slang" pixelMain)
    rachis_rhi_slang_spirv(BootstrapCube.cs.spv "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapCube.slang" computeMain)
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanSpecialization.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanSpecialization.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanSpecialization"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanGraphicsTriangle.hlsl"
        ENTRY directVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.direct.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanGraphicsTriangle.direct.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanGraphicsTriangle.hlsl"
        ENTRY inputVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.input.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanGraphicsTriangle.input.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanGraphicsTriangle.hlsl"
        ENTRY depthBiasVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.depth.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanGraphicsTriangle.depth.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanGraphicsTriangle.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanGraphicsTriangle.ps"
    )
    set(rachis_stage_linkage_spirv)
    foreach(rachis_stage_linkage_entry IN ITEMS
            float4Vertex:vs float2Vertex:vs uint4Vertex:vs layoutVertex:vs
            float4Pixel:ps float2Pixel:ps location1Pixel:ps)
        string(REPLACE ":" ";" rachis_stage_linkage_parts "${rachis_stage_linkage_entry}")
        list(GET rachis_stage_linkage_parts 0 rachis_stage_linkage_name)
        list(GET rachis_stage_linkage_parts 1 rachis_stage_linkage_kind)
        rachis_cook_dxc_spirv(
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanStageLinkage.hlsl"
            ENTRY ${rachis_stage_linkage_name}
            PROFILE ${rachis_stage_linkage_kind}_6_0
            OUTPUT "${rachis_rhi_spirv_dir}/VulkanStageLinkage.${rachis_stage_linkage_name}.spv"
            COMMENT "Cooking Vulkan SPIR-V VulkanStageLinkage.${rachis_stage_linkage_name}"
        )
        list(APPEND rachis_stage_linkage_spirv
            "${rachis_rhi_spirv_dir}/VulkanStageLinkage.${rachis_stage_linkage_name}.spv")
    endforeach()
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl"
        ENTRY matrixVertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VertexInput.matrix.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VertexInput.matrix.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl"
        ENTRY arrayVertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VertexInput.array.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VertexInput.array.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VertexInput.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VertexInput.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanPushConstantRanges.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.low.vs.spv"
        DEFINES RACHIS_PUSH_VERTEX_LOW
        COMMENT "Cooking Vulkan SPIR-V VulkanPushConstantRanges.low.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanPushConstantRanges.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.high.vs.spv"
        DEFINES RACHIS_PUSH_VERTEX_HIGH
        COMMENT "Cooking Vulkan SPIR-V VulkanPushConstantRanges.high.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanPushConstantRanges.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.contiguous.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanPushConstantRanges.contiguous.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanPushConstantRanges.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.split.ps.spv"
        DEFINES RACHIS_PUSH_PIXEL_SPLIT
        COMMENT "Cooking Vulkan SPIR-V VulkanPushConstantRanges.split.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanHeadlessParity.hlsl"
        ENTRY texturedVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanHeadlessParity.textured.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanHeadlessParity.textured.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanHeadlessParity.hlsl"
        ENTRY texturedPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanHeadlessParity.textured.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanHeadlessParity.textured.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PartiallyBoundBinding.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/PartiallyBoundBinding.spv"
        COMMENT "Cooking Vulkan SPIR-V PartiallyBoundBinding"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VariableBinding.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VariableBinding.spv"
        COMMENT "Cooking Vulkan SPIR-V VariableBinding"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/UpdateAfterBindBinding.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/UpdateAfterBindBinding.spv"
        COMMENT "Cooking Vulkan SPIR-V UpdateAfterBindBinding"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/Shader16Bit.hlsl"
        ENTRY main
        PROFILE cs_6_2
        OUTPUT "${rachis_rhi_spirv_dir}/Shader16Bit.spv"
        OPTIONS -enable-16bit-types
        COMMENT "Cooking Vulkan SPIR-V Shader16Bit"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/ShaderInt64.spv"
        COMMENT "Cooking Vulkan SPIR-V ShaderInt64"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64AtomicGroupShared.hlsl"
        ENTRY main
        PROFILE cs_6_6
        OUTPUT "${rachis_rhi_spirv_dir}/ShaderInt64AtomicGroupShared.spv"
        COMMENT "Cooking Vulkan SPIR-V ShaderInt64AtomicGroupShared"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64AtomicTypedResource.hlsl"
        ENTRY main
        PROFILE cs_6_6
        OUTPUT "${rachis_rhi_spirv_dir}/ShaderInt64AtomicTypedResource.spv"
        COMMENT "Cooking Vulkan SPIR-V ShaderInt64AtomicTypedResource"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64AtomicStorageBuffer.hlsl"
        ENTRY main
        PROFILE cs_6_6
        OUTPUT "${rachis_rhi_spirv_dir}/ShaderInt64AtomicStorageBuffer.spv"
        COMMENT "Cooking Vulkan SPIR-V ShaderInt64AtomicStorageBuffer"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SubgroupOperations.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/SubgroupOperations.spv"
        COMMENT "Cooking Vulkan SPIR-V SubgroupOperations"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanComputeWrite.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanComputeWrite.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanComputeWrite"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanComputeWrite.hlsl"
        ENTRY scale
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanComputeWrite.scale.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanComputeWrite.scale"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanComputeWrite.hlsl"
        ENTRY countDispatch
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanComputeWrite.countDispatch.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanComputeWrite.countDispatch"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ConditionalExecution.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/ConditionalExecution.spv"
        COMMENT "Cooking Vulkan SPIR-V ConditionalExecution"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DynamicBindingCounter.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/DynamicBindingCounter.spv"
        COMMENT "Cooking Vulkan SPIR-V DynamicBindingCounter"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/UniformBufferRange.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/UniformBufferRange.spv"
        COMMENT "Cooking Vulkan SPIR-V UniformBufferRange"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DynamicUniformBufferArray.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/DynamicUniformBufferArray.spv"
        COMMENT "Cooking Vulkan SPIR-V DynamicUniformBufferArray"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanComputeImage.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanComputeImage.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanComputeImage"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ReadOnlyStorageBuffers.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/ReadOnlyStorageBuffers.spv"
        DEFINES RACHIS_VULKAN=1
        COMMENT "Cooking Vulkan SPIR-V ReadOnlyStorageBuffers"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanSampledTexture.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanSampledTexture.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanSampledTexture"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanIndirectArguments.hlsl"
        ENTRY writeDraw
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanIndirectArguments.draw.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanIndirectArguments.draw"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanIndirectArguments.hlsl"
        ENTRY writeIndexedDraw
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanIndirectArguments.indexed.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanIndirectArguments.indexed"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanIndirectArguments.hlsl"
        ENTRY writeDispatch
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanIndirectArguments.dispatch.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanIndirectArguments.dispatch"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl"
        ENTRY pointVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GeometryShader.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V GeometryShader.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl"
        ENTRY geometryResourceMain
        PROFILE gs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GeometryShader.gs.spv"
        COMMENT "Cooking Vulkan SPIR-V GeometryShader.gs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl"
        ENTRY viewportArrayMain
        PROFILE gs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GeometryShader.viewport.gs.spv"
        COMMENT "Cooking Vulkan SPIR-V GeometryShader.viewport.gs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl"
        ENTRY geometryPixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/GeometryShader.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V GeometryShader.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl"
        ENTRY tessellationVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TessellationShader.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V TessellationShader.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl"
        ENTRY tessellationHull
        PROFILE hs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TessellationShader.hs.spv"
        COMMENT "Cooking Vulkan SPIR-V TessellationShader.hs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl"
        ENTRY tessellationDomain
        PROFILE ds_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TessellationShader.ds.spv"
        COMMENT "Cooking Vulkan SPIR-V TessellationShader.ds"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl"
        ENTRY tessellationPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TessellationShader.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V TessellationShader.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LayeredRendering.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/LayeredRendering.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V LayeredRendering.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LayeredRendering.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/LayeredRendering.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V LayeredRendering.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MultiviewRendering.hlsl"
        ENTRY vsMain
        PROFILE vs_6_1
        OUTPUT "${rachis_rhi_spirv_dir}/MultiviewRendering.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V MultiviewRendering.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MultiviewRendering.hlsl"
        ENTRY psMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/MultiviewRendering.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V MultiviewRendering.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderDistance.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/ShaderDistance.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V ShaderDistance.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderDistance.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/ShaderDistance.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V ShaderDistance.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl"
        ENTRY primitiveIdVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.primitive.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V FragmentShaderBuiltins.primitive.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl"
        ENTRY primitiveIdPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.primitive.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V FragmentShaderBuiltins.primitive.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl"
        ENTRY barycentricVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.barycentric.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V FragmentShaderBuiltins.barycentric.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl"
        ENTRY barycentricPixel
        PROFILE ps_6_1
        OUTPUT "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.barycentric.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V FragmentShaderBuiltins.barycentric.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DualSourceBlend.hlsl"
        ENTRY main
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/DualSourceBlend.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V DualSourceBlend.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LogicOperation.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/LogicOperation.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V LogicOperation.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanRasterizerOrderedStorage.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanRasterizerOrderedStorage.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanRasterizerOrderedStorage.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RasterizerOrderedStorage.hlsl"
        ENTRY main
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/RasterizerOrderedStorage.ps.spv"
        OPTIONS -fspv-extension=SPV_EXT_fragment_shader_interlock
        COMMENT "Cooking Vulkan SPIR-V RasterizerOrderedStorage.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY fullscreenVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.fullscreen.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.fullscreen.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY storeVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.store.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.store.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY pulledVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.pulled.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.pulled.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY solidPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.solid.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.solid.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY sampleIndexPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.sample-index.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.sample-index.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY minLodPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.min-lod.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.min-lod.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanBaseShaderFeatures.hlsl"
        ENTRY gatherOffsetPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.gather-offset.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanBaseShaderFeatures.gather-offset.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanShaderTypeFeatures.hlsl"
        ENTRY halfStorageCompute
        PROFILE cs_6_2
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.half-storage.cs.spv"
        OPTIONS -enable-16bit-types
        COMMENT "Cooking Vulkan SPIR-V VulkanShaderTypeFeatures.half-storage.cs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanShaderTypeFeatures.hlsl"
        ENTRY doubleCompute
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.double.cs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanShaderTypeFeatures.double.cs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanShaderTypeFeatures.hlsl"
        ENTRY halfStageVertex
        PROFILE vs_6_2
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.half-stage.vs.spv"
        OPTIONS -enable-16bit-types
        COMMENT "Cooking Vulkan SPIR-V VulkanShaderTypeFeatures.half-stage.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanShaderTypeFeatures.hlsl"
        ENTRY halfStagePixel
        PROFILE ps_6_2
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.half-stage.ps.spv"
        OPTIONS -enable-16bit-types
        COMMENT "Cooking Vulkan SPIR-V VulkanShaderTypeFeatures.half-stage.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanShaderTypeFeatures.hlsl"
        ENTRY floatStageVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.float-stage.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanShaderTypeFeatures.float-stage.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VulkanShaderTypeFeatures.hlsl"
        ENTRY floatStagePixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.float-stage.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V VulkanShaderTypeFeatures.float-stage.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SamplerReduction.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/SamplerReduction.spv"
        COMMENT "Cooking Vulkan SPIR-V SamplerReduction"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PrimitiveRestart.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/PrimitiveRestart.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V PrimitiveRestart.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PrimitiveRestart.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/PrimitiveRestart.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V PrimitiveRestart.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TriangleFan.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TriangleFan.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V TriangleFan.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TriangleFan.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TriangleFan.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V TriangleFan.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PointList.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/PointList.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V PointList.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PointList.hlsl"
        ENTRY sizedVertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/PointList.sized.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V PointList.sized.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PointList.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/PointList.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V PointList.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SamplerFilter.hlsl"
        ENTRY vertexMain
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/SamplerFilter.vs.spv"
        COMMENT "Cooking Vulkan SPIR-V SamplerFilter.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SamplerFilter.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/SamplerFilter.ps.spv"
        COMMENT "Cooking Vulkan SPIR-V SamplerFilter.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VariableRateShadingWorkload.hlsl"
        ENTRY directVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VariableRateShadingWorkload.direct.vs.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V VariableRateShadingWorkload.direct.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VariableRateShadingWorkload.hlsl"
        ENTRY shadingRateVertex
        PROFILE vs_6_5
        OUTPUT "${rachis_rhi_spirv_dir}/VariableRateShadingWorkload.rate.vs.spv"
        OPTIONS -fspv-extension=SPV_KHR_fragment_shading_rate
        COMMENT "Cooking RHI test Vulkan SPIR-V VariableRateShadingWorkload.rate.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VariableRateShadingWorkload.hlsl"
        ENTRY pixelMain
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/VariableRateShadingWorkload.ps.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V VariableRateShadingWorkload.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TaskMeshWorkload.hlsl"
        ENTRY taskMain
        PROFILE as_6_5
        OUTPUT "${rachis_rhi_spirv_dir}/TaskMeshWorkload.as.spv"
        OPTIONS -fspv-extension=SPV_EXT_mesh_shader
        COMMENT "Cooking RHI test Vulkan SPIR-V TaskMeshWorkload.as"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TaskMeshWorkload.hlsl"
        ENTRY meshMain
        PROFILE ms_6_5
        OUTPUT "${rachis_rhi_spirv_dir}/TaskMeshWorkload.ms.spv"
        OPTIONS -fspv-extension=SPV_EXT_mesh_shader
        COMMENT "Cooking RHI test Vulkan SPIR-V TaskMeshWorkload.ms"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TaskMeshWorkload.hlsl"
        ENTRY meshNoTask
        PROFILE ms_6_5
        OUTPUT "${rachis_rhi_spirv_dir}/TaskMeshWorkload.no-task.ms.spv"
        OPTIONS -fspv-extension=SPV_EXT_mesh_shader
        COMMENT "Cooking RHI test Vulkan SPIR-V TaskMeshWorkload.no-task.ms"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TaskMeshWorkload.hlsl"
        ENTRY meshPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/TaskMeshWorkload.ps.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V TaskMeshWorkload.ps"
    )
    foreach(rachis_draw_index_entry IN ITEMS
            drawIndexVertex:vs_6_0:DrawIndex.vs plainVertex:vs_6_0:DrawIndex.plain.vs
            drawIndexMesh:ms_6_5:DrawIndex.ms recordPixel:ps_6_0:DrawIndex.ps)
        string(REPLACE ":" ";" rachis_draw_index_parts "${rachis_draw_index_entry}")
        list(GET rachis_draw_index_parts 0 rachis_draw_index_name)
        list(GET rachis_draw_index_parts 1 rachis_draw_index_profile)
        list(GET rachis_draw_index_parts 2 rachis_draw_index_output)
        set(rachis_draw_index_options)
        if(rachis_draw_index_profile STREQUAL "ms_6_5")
            set(rachis_draw_index_options -fspv-extension=SPV_EXT_mesh_shader)
        endif()
        rachis_cook_dxc_spirv(
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DrawIndex.hlsl"
            ENTRY ${rachis_draw_index_name}
            PROFILE ${rachis_draw_index_profile}
            OUTPUT "${rachis_rhi_spirv_dir}/${rachis_draw_index_output}.spv"
            OPTIONS ${rachis_draw_index_options}
            INCLUDE_DIRECTORIES "${rachis_rhi_public_include_dir}"
            COMMENT "Cooking RHI test Vulkan SPIR-V ${rachis_draw_index_output}"
        )
    endforeach()
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RayQueryWorkload.hlsl"
        ENTRY main
        PROFILE cs_6_5
        OUTPUT "${rachis_rhi_spirv_dir}/RayQueryWorkload.spv"
        OPTIONS -fspv-extension=SPV_KHR_ray_query
        COMMENT "Cooking RHI test Vulkan SPIR-V RayQueryWorkload"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/InstanceRecordWriter.hlsl"
        ENTRY main
        PROFILE cs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/InstanceRecordWriter.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V InstanceRecordWriter"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/InstanceBufferRayQuery.hlsl"
        ENTRY main
        PROFILE cs_6_5
        OUTPUT "${rachis_rhi_spirv_dir}/InstanceBufferRayQuery.spv"
        OPTIONS -fspv-extension=SPV_KHR_ray_query
        COMMENT "Cooking RHI test Vulkan SPIR-V InstanceBufferRayQuery"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RayTracingPipeline.hlsl"
        ENTRY rayGenerationWithoutRecord
        PROFILE lib_6_3
        OUTPUT "${rachis_rhi_spirv_dir}/RayTracingPipeline.raygen.spv"
        OPTIONS -fspv-extension=SPV_KHR_ray_tracing
        COMMENT "Cooking Vulkan SPIR-V RayTracingPipeline.raygen"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RayTracingPipeline.hlsl"
        ENTRY missShader
        PROFILE lib_6_3
        OUTPUT "${rachis_rhi_spirv_dir}/RayTracingPipeline.miss.spv"
        OPTIONS -fspv-extension=SPV_KHR_ray_tracing
        COMMENT "Cooking Vulkan SPIR-V RayTracingPipeline.miss"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RayTracingPushConstants.hlsl"
        ENTRY rayGenerationWithPushConstants
        PROFILE lib_6_3
        OUTPUT "${rachis_rhi_spirv_dir}/RayTracingPushConstants.raygen.spv"
        OPTIONS -fspv-extension=SPV_KHR_ray_tracing
        COMMENT "Cooking Vulkan SPIR-V RayTracingPushConstants.raygen"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl"
        ENTRY fullscreenVertex
        PROFILE vs_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/MipmapAfterRendering.vs.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V MipmapAfterRendering.vs"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl"
        ENTRY slowPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/MipmapAfterRendering.slow.ps.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V MipmapAfterRendering.slow.ps"
    )
    rachis_cook_dxc_spirv(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl"
        ENTRY blendPixel
        PROFILE ps_6_0
        OUTPUT "${rachis_rhi_spirv_dir}/MipmapAfterRendering.blend.ps.spv"
        COMMENT "Cooking RHI test Vulkan SPIR-V MipmapAfterRendering.blend.ps"
    )
    add_custom_target(RachisRHITestSpirvShaders ALL
        DEPENDS
            "${rachis_rhi_spirv_dir}/BootstrapTriangle.vs.spv"
            "${rachis_rhi_spirv_dir}/BootstrapTriangle.ps.spv"
            "${rachis_rhi_spirv_dir}/BootstrapCube.vs.spv"
            "${rachis_rhi_spirv_dir}/BootstrapCube.ps.spv"
            "${rachis_rhi_spirv_dir}/BootstrapCube.cs.spv"
            "${rachis_rhi_spirv_dir}/VulkanSpecialization.spv"
            "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.direct.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.input.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.depth.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanGraphicsTriangle.ps.spv"
            ${rachis_stage_linkage_spirv}
            "${rachis_rhi_spirv_dir}/VertexInput.matrix.vs.spv"
            "${rachis_rhi_spirv_dir}/VertexInput.array.vs.spv"
            "${rachis_rhi_spirv_dir}/VertexInput.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.low.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.high.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.contiguous.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanPushConstantRanges.split.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanHeadlessParity.textured.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanHeadlessParity.textured.ps.spv"
            "${rachis_rhi_spirv_dir}/PartiallyBoundBinding.spv"
            "${rachis_rhi_spirv_dir}/VariableBinding.spv"
            "${rachis_rhi_spirv_dir}/UpdateAfterBindBinding.spv"
            "${rachis_rhi_spirv_dir}/Shader16Bit.spv"
            "${rachis_rhi_spirv_dir}/ShaderInt64.spv"
            "${rachis_rhi_spirv_dir}/ShaderInt64AtomicGroupShared.spv"
            "${rachis_rhi_spirv_dir}/ShaderInt64AtomicTypedResource.spv"
            "${rachis_rhi_spirv_dir}/ShaderInt64AtomicStorageBuffer.spv"
            "${rachis_rhi_spirv_dir}/SubgroupOperations.spv"
            "${rachis_rhi_spirv_dir}/VulkanComputeWrite.spv"
            "${rachis_rhi_spirv_dir}/VulkanComputeWrite.scale.spv"
            "${rachis_rhi_spirv_dir}/VulkanComputeWrite.countDispatch.spv"
            "${rachis_rhi_spirv_dir}/ConditionalExecution.spv"
            "${rachis_rhi_spirv_dir}/DynamicBindingCounter.spv"
            "${rachis_rhi_spirv_dir}/UniformBufferRange.spv"
            "${rachis_rhi_spirv_dir}/DynamicUniformBufferArray.spv"
            "${rachis_rhi_spirv_dir}/VulkanComputeImage.spv"
            "${rachis_rhi_spirv_dir}/ReadOnlyStorageBuffers.spv"
            "${rachis_rhi_spirv_dir}/VulkanSampledTexture.spv"
            "${rachis_rhi_spirv_dir}/VulkanIndirectArguments.draw.spv"
            "${rachis_rhi_spirv_dir}/VulkanIndirectArguments.indexed.spv"
            "${rachis_rhi_spirv_dir}/VulkanIndirectArguments.dispatch.spv"
            "${rachis_rhi_spirv_dir}/GeometryShader.vs.spv"
            "${rachis_rhi_spirv_dir}/GeometryShader.gs.spv"
            "${rachis_rhi_spirv_dir}/GeometryShader.viewport.gs.spv"
            "${rachis_rhi_spirv_dir}/GeometryShader.ps.spv"
            "${rachis_rhi_spirv_dir}/TessellationShader.vs.spv"
            "${rachis_rhi_spirv_dir}/TessellationShader.hs.spv"
            "${rachis_rhi_spirv_dir}/TessellationShader.ds.spv"
            "${rachis_rhi_spirv_dir}/TessellationShader.ps.spv"
            "${rachis_rhi_spirv_dir}/LayeredRendering.vs.spv"
            "${rachis_rhi_spirv_dir}/LayeredRendering.ps.spv"
            "${rachis_rhi_spirv_dir}/MultiviewRendering.vs.spv"
            "${rachis_rhi_spirv_dir}/MultiviewRendering.ps.spv"
            "${rachis_rhi_spirv_dir}/ShaderDistance.vs.spv"
            "${rachis_rhi_spirv_dir}/ShaderDistance.ps.spv"
            "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.primitive.vs.spv"
            "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.primitive.ps.spv"
            "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.barycentric.vs.spv"
            "${rachis_rhi_spirv_dir}/FragmentShaderBuiltins.barycentric.ps.spv"
            "${rachis_rhi_spirv_dir}/DualSourceBlend.ps.spv"
            "${rachis_rhi_spirv_dir}/LogicOperation.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanRasterizerOrderedStorage.vs.spv"
            "${rachis_rhi_spirv_dir}/RasterizerOrderedStorage.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.fullscreen.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.store.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.pulled.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.solid.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.sample-index.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.min-lod.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanBaseShaderFeatures.gather-offset.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.half-storage.cs.spv"
            "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.double.cs.spv"
            "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.half-stage.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.half-stage.ps.spv"
            "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.float-stage.vs.spv"
            "${rachis_rhi_spirv_dir}/VulkanShaderTypeFeatures.float-stage.ps.spv"
            "${rachis_rhi_spirv_dir}/SamplerReduction.spv"
            "${rachis_rhi_spirv_dir}/PrimitiveRestart.vs.spv"
            "${rachis_rhi_spirv_dir}/PrimitiveRestart.ps.spv"
            "${rachis_rhi_spirv_dir}/TriangleFan.vs.spv"
            "${rachis_rhi_spirv_dir}/TriangleFan.ps.spv"
            "${rachis_rhi_spirv_dir}/PointList.vs.spv"
            "${rachis_rhi_spirv_dir}/PointList.sized.vs.spv"
            "${rachis_rhi_spirv_dir}/PointList.ps.spv"
            "${rachis_rhi_spirv_dir}/SamplerFilter.vs.spv"
            "${rachis_rhi_spirv_dir}/SamplerFilter.ps.spv"
            "${rachis_rhi_spirv_dir}/VariableRateShadingWorkload.direct.vs.spv"
            "${rachis_rhi_spirv_dir}/VariableRateShadingWorkload.rate.vs.spv"
            "${rachis_rhi_spirv_dir}/VariableRateShadingWorkload.ps.spv"
            "${rachis_rhi_spirv_dir}/TaskMeshWorkload.as.spv"
            "${rachis_rhi_spirv_dir}/TaskMeshWorkload.ms.spv"
            "${rachis_rhi_spirv_dir}/TaskMeshWorkload.no-task.ms.spv"
            "${rachis_rhi_spirv_dir}/TaskMeshWorkload.ps.spv"
            "${rachis_rhi_spirv_dir}/DrawIndex.vs.spv"
            "${rachis_rhi_spirv_dir}/DrawIndex.plain.vs.spv"
            "${rachis_rhi_spirv_dir}/DrawIndex.ms.spv"
            "${rachis_rhi_spirv_dir}/DrawIndex.ps.spv"
            "${rachis_rhi_spirv_dir}/RayQueryWorkload.spv"
            "${rachis_rhi_spirv_dir}/InstanceRecordWriter.spv"
            "${rachis_rhi_spirv_dir}/InstanceBufferRayQuery.spv"
            "${rachis_rhi_spirv_dir}/RayTracingPipeline.raygen.spv"
            "${rachis_rhi_spirv_dir}/RayTracingPipeline.miss.spv"
            "${rachis_rhi_spirv_dir}/RayTracingPushConstants.raygen.spv"
            "${rachis_rhi_spirv_dir}/MipmapAfterRendering.vs.spv"
            "${rachis_rhi_spirv_dir}/MipmapAfterRendering.slow.ps.spv"
            "${rachis_rhi_spirv_dir}/MipmapAfterRendering.blend.ps.spv"
    )
endif()

if(RACHIS_ENABLE_METAL)
    set(rachis_rhi_metallib_dir "${CMAKE_CURRENT_BINARY_DIR}/Generated/metallib")
    set(rachis_mipmaps_hlsl "${CMAKE_CURRENT_SOURCE_DIR}/src/D3D12/Shaders/GenerateMipmaps.hlsl")
    function(rachis_rhi_slang_metallib name source entry)
        cmake_parse_arguments(METAL "NO_WARNINGS_AS_ERRORS" "" "DEFINES" ${ARGN})
        if(METAL_NO_WARNINGS_AS_ERRORS)
            rachis_cook_slang_metallib(
                SOURCE "${source}"
                ENTRY "${entry}"
                OUTPUT "${rachis_rhi_metallib_dir}/${name}"
                DEFINES ${METAL_DEFINES}
                NO_WARNINGS_AS_ERRORS
                COMMENT "Cooking Metal metallib ${name}"
            )
        else()
            rachis_cook_slang_metallib(
                SOURCE "${source}"
                ENTRY "${entry}"
                OUTPUT "${rachis_rhi_metallib_dir}/${name}"
                DEFINES ${METAL_DEFINES}
                COMMENT "Cooking Metal metallib ${name}"
            )
        endif()
    endfunction()
    rachis_rhi_slang_metallib(GenerateMipmaps2D.metallib "${rachis_mipmaps_hlsl}" main NO_WARNINGS_AS_ERRORS DEFINES RACHIS_MIPMAP_DIMENSION=2)
    rachis_rhi_slang_metallib(GenerateMipmaps3D.metallib "${rachis_mipmaps_hlsl}" main NO_WARNINGS_AS_ERRORS DEFINES RACHIS_MIPMAP_DIMENSION=3)
    rachis_ensure_shader_binary_tool(rachis_shader_binary_tool_cmd rachis_shader_binary_tool_depends)
    function(rachis_embed_pnms name symbol)
        add_custom_command(
            OUTPUT "${rachis_rhi_metallib_dir}/Rachis${name}.pnms.h"
            COMMAND
                "${rachis_shader_binary_tool_cmd}"
                embed-header
                --input "${rachis_rhi_metallib_dir}/${name}.pnms"
                --output "${rachis_rhi_metallib_dir}/Rachis${name}.pnms.h"
                --symbol "${symbol}"
            DEPENDS
                "${rachis_rhi_metallib_dir}/${name}.pnms"
                ${rachis_shader_binary_tool_depends}
            COMMENT "Embedding Metal ${name}.pnms"
            VERBATIM
        )
    endfunction()
    rachis_embed_pnms(GenerateMipmaps2D g_rachisGenerateMipmaps2DPnms)
    rachis_embed_pnms(GenerateMipmaps3D g_rachisGenerateMipmaps3DPnms)
    add_custom_target(RachisRHIMetalShaders ALL
        DEPENDS
            "${rachis_rhi_metallib_dir}/GenerateMipmaps2D.metallib"
            "${rachis_rhi_metallib_dir}/GenerateMipmaps2D.pnms"
            "${rachis_rhi_metallib_dir}/RachisGenerateMipmaps2D.pnms.h"
            "${rachis_rhi_metallib_dir}/GenerateMipmaps3D.metallib"
            "${rachis_rhi_metallib_dir}/GenerateMipmaps3D.pnms"
            "${rachis_rhi_metallib_dir}/RachisGenerateMipmaps3D.pnms.h"
    )
endif()

if(RACHIS_ENABLE_METAL AND rachis_build_tests)
    set(rachis_layout_fixture "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LayoutFixture.slang")
    rachis_rhi_slang_metallib(BootstrapTriangle.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapTriangle.slang" vertexMain)
    rachis_rhi_slang_metallib(BootstrapTriangle.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapTriangle.slang" pixelMain)
    rachis_rhi_slang_metallib(BootstrapCube.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapCube.slang" vertexMain)
    rachis_rhi_slang_metallib(BootstrapCube.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapCube.slang" pixelMain)
    rachis_rhi_slang_metallib(BootstrapCube.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BootstrapCube.slang" computeMain)
    rachis_rhi_slang_metallib(LayoutFixture.cs.metallib "${rachis_layout_fixture}" computeMain)
    rachis_rhi_slang_metallib(MetalUnusedBindings.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalUnusedBindings.slang" vertexMain)
    rachis_rhi_slang_metallib(MetalSpecialization.7.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalSpecialization.slang" computeMain DEFINES RACHIS_SPECIALIZED_VALUE=7)
    rachis_rhi_slang_metallib(MetalSpecialization.13.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalSpecialization.slang" computeMain DEFINES RACHIS_SPECIALIZED_VALUE=13)
    rachis_rhi_slang_metallib(MetalPushConstant.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalPushConstant.slang" computeMain)
    rachis_rhi_slang_metallib(MetalPushConstant.set1.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalPushConstant.slang" computeMain DEFINES RACHIS_OUTPUT_SPACE=space1)
    rachis_rhi_slang_metallib(MetalPushConstant.set3.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalPushConstant.slang" computeMain DEFINES RACHIS_OUTPUT_SPACE=space3)
    rachis_rhi_slang_metallib(MetalUnusedImageBindings.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalUnusedImageBindings.slang" vertexMain)
    rachis_rhi_slang_metallib(MetalRasterInstance.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRasterInstance.slang" vertexMain)
    rachis_rhi_slang_metallib(MetalRasterInstance.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRasterInstance.slang" pixelMain)
    rachis_rhi_slang_metallib(MetalRasterVertex.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRasterVertex.slang" vertexMain)
    rachis_rhi_slang_metallib(MetalRasterVertex.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRasterVertex.slang" pixelMain)
    rachis_rhi_slang_metallib(MetalStageLinkage.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" vertexMain)
    rachis_rhi_slang_metallib(MetalStageLinkage.matching.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" matchingPixel)
    rachis_rhi_slang_metallib(MetalStageLinkage.fewer.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" fewerComponentsPixel)
    rachis_rhi_slang_metallib(MetalStageLinkage.more.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" moreComponentsPixel)
    rachis_rhi_slang_metallib(MetalStageLinkage.type.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" typeMismatchPixel)
    rachis_rhi_slang_metallib(MetalStageLinkage.missing.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" missingLocationPixel)
    rachis_rhi_slang_metallib(MetalStageLinkage.renamed.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalStageLinkage.slang" renamedPixel)
    rachis_rhi_slang_metallib(MetalComputeStorage.scatterMain.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalComputeStorage.slang" scatterMain)
    rachis_rhi_slang_metallib(MetalComputeStorage.reduceMain.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalComputeStorage.slang" reduceMain)
    rachis_rhi_slang_metallib(MetalShaderArithmetic.arithmetic16Main.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalShaderArithmetic.slang" arithmetic16Main)
    rachis_rhi_slang_metallib(MetalShaderArithmetic.int64Main.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalShaderArithmetic.slang" int64Main)
    rachis_rhi_slang_metallib(MetalShaderArithmetic.subgroupMain.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalShaderArithmetic.slang" subgroupMain)
    rachis_rhi_slang_metallib(MetalLayeredRendering.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalLayeredRendering.slang" vertexMain)
    rachis_rhi_slang_metallib(MetalLayeredRendering.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalLayeredRendering.slang" pixelMain)
    rachis_rhi_slang_metallib(MetalRayQuery.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRayQuery.slang" computeMain)
    rachis_rhi_slang_metallib(MetalHeadlessParity.direct.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalHeadlessParity.slang" directVertex DEFINES RACHIS_PARITY_VERTEX=1)
    rachis_rhi_slang_metallib(MetalHeadlessParity.depth.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalHeadlessParity.slang" depthBiasVertex DEFINES RACHIS_PARITY_VERTEX=1)
    rachis_rhi_slang_metallib(MetalHeadlessParity.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalHeadlessParity.slang" pixelMain DEFINES RACHIS_PARITY_PIXEL=1)
    rachis_rhi_slang_metallib(MetalHeadlessParity.textured.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalHeadlessParity.slang" texturedVertex DEFINES RACHIS_PARITY_TEXTURED_VERTEX=1)
    rachis_rhi_slang_metallib(MetalHeadlessParity.textured.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalHeadlessParity.slang" texturedPixel DEFINES RACHIS_PARITY_TEXTURED=1)
    rachis_rhi_slang_metallib(MetalHeadlessParity.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalHeadlessParityCompute.slang" computeMain)
    rachis_rhi_slang_metallib(MetalRemapContract.single.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRemapContract.slang" pushAndBindingZeroCompute DEFINES RACHIS_REMAP_SINGLE_COMPUTE=1)
    rachis_rhi_slang_metallib(MetalRemapContract.multi.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRemapContract.slang" multiSetCompute DEFINES RACHIS_REMAP_MULTI_COMPUTE=1)
    rachis_rhi_slang_metallib(MetalRemapContract.single.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRemapContract.slang" pushAndBindingZeroVertex DEFINES RACHIS_REMAP_SINGLE_GRAPHICS_VS=1)
    rachis_rhi_slang_metallib(MetalRemapContract.single.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRemapContract.slang" pushAndBindingZeroPixel DEFINES RACHIS_REMAP_SINGLE_GRAPHICS_PS=1)
    rachis_rhi_slang_metallib(MetalRemapContract.multi.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRemapContract.slang" multiSetVertex DEFINES RACHIS_REMAP_MULTI_GRAPHICS_VS=1)
    rachis_rhi_slang_metallib(MetalRemapContract.multi.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalRemapContract.slang" multiSetPixel DEFINES RACHIS_REMAP_MULTI_GRAPHICS_PS=1)
    rachis_rhi_slang_metallib(MipmapAfterRendering.vs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl" fullscreenVertex)
    rachis_rhi_slang_metallib(MipmapAfterRendering.slow.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl" slowPixel)
    rachis_rhi_slang_metallib(MipmapAfterRendering.blend.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl" blendPixel)
    rachis_rhi_slang_metallib(MetalMeshShader.as.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalMeshShader.slang" taskMain)
    rachis_rhi_slang_metallib(MetalMeshShader.ms.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalMeshShader.slang" meshMain)
    rachis_rhi_slang_metallib(MetalMeshShader.no-task.ms.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalMeshShader.slang" meshNoTask)
    rachis_rhi_slang_metallib(MetalMeshShader.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalMeshShader.slang" meshPixel)
    rachis_rhi_slang_metallib(MetalMeshShader.missing.ps.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalMeshShader.slang" meshPixelMissing)
    rachis_rhi_slang_metallib(MetalIndirectCount.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalIndirectCount.slang" countMain)
    rachis_rhi_slang_metallib(MetalBindingArrays.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalBindingArrays.slang" bindingArraysMain)
    rachis_rhi_slang_metallib(MetalUpdateAfterBind.cs.metallib "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MetalUpdateAfterBind.slang" updateAfterBindMain)
    rachis_cook_slang_metallib(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DrawIndex.hlsl"
        ENTRY drawIndexVertex
        STAGE vertex
        OUTPUT "${rachis_rhi_metallib_dir}/DrawIndex.vs.metallib"
        INCLUDE_DIRECTORIES "${rachis_rhi_public_include_dir}"
        COMMENT "Cooking Metal metallib DrawIndex.vs"
    )
    rachis_cook_slang_metallib(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DrawIndex.hlsl"
        ENTRY plainVertex
        STAGE vertex
        OUTPUT "${rachis_rhi_metallib_dir}/DrawIndex.plain.vs.metallib"
        INCLUDE_DIRECTORIES "${rachis_rhi_public_include_dir}"
        COMMENT "Cooking Metal metallib DrawIndex.plain.vs"
    )
    rachis_cook_slang_metallib(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DrawIndex.hlsl"
        ENTRY drawIndexMesh
        STAGE mesh
        OUTPUT "${rachis_rhi_metallib_dir}/DrawIndex.ms.metallib"
        INCLUDE_DIRECTORIES "${rachis_rhi_public_include_dir}"
        COMMENT "Cooking Metal metallib DrawIndex.ms"
    )
    rachis_cook_slang_metallib(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DrawIndex.hlsl"
        ENTRY recordPixel
        STAGE pixel
        OUTPUT "${rachis_rhi_metallib_dir}/DrawIndex.ps.metallib"
        INCLUDE_DIRECTORIES "${rachis_rhi_public_include_dir}"
        COMMENT "Cooking Metal metallib DrawIndex.ps"
    )
    rachis_cook_slang_metallib(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl"
        ENTRY inputVertex
        STAGE vertex
        OUTPUT "${rachis_rhi_metallib_dir}/GraphicsTriangle.input.vs.metallib"
        COMMENT "Cooking Metal metallib GraphicsTriangle.input.vs"
    )
    rachis_cook_slang_metallib(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl"
        ENTRY pixelMain
        STAGE pixel
        OUTPUT "${rachis_rhi_metallib_dir}/GraphicsTriangle.ps.metallib"
        COMMENT "Cooking Metal metallib GraphicsTriangle.ps"
    )
    add_custom_target(RachisRHITestMetalShaders ALL
        DEPENDS
            "${rachis_rhi_metallib_dir}/MetalMeshShader.as.metallib"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.as.pnms"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.ms.metallib"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.ms.pnms"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.no-task.ms.metallib"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.no-task.ms.pnms"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.missing.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalMeshShader.missing.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalIndirectCount.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalIndirectCount.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalBindingArrays.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalBindingArrays.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalUpdateAfterBind.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalUpdateAfterBind.cs.pnms"
            "${rachis_rhi_metallib_dir}/DrawIndex.vs.metallib"
            "${rachis_rhi_metallib_dir}/DrawIndex.vs.pnms"
            "${rachis_rhi_metallib_dir}/DrawIndex.plain.vs.metallib"
            "${rachis_rhi_metallib_dir}/DrawIndex.plain.vs.pnms"
            "${rachis_rhi_metallib_dir}/DrawIndex.ms.metallib"
            "${rachis_rhi_metallib_dir}/DrawIndex.ms.pnms"
            "${rachis_rhi_metallib_dir}/DrawIndex.ps.metallib"
            "${rachis_rhi_metallib_dir}/DrawIndex.ps.pnms"
            "${rachis_rhi_metallib_dir}/GraphicsTriangle.input.vs.metallib"
            "${rachis_rhi_metallib_dir}/GraphicsTriangle.input.vs.pnms"
            "${rachis_rhi_metallib_dir}/GraphicsTriangle.ps.metallib"
            "${rachis_rhi_metallib_dir}/GraphicsTriangle.ps.pnms"
            "${rachis_rhi_metallib_dir}/BootstrapTriangle.vs.metallib"
            "${rachis_rhi_metallib_dir}/BootstrapTriangle.vs.pnms"
            "${rachis_rhi_metallib_dir}/BootstrapTriangle.ps.metallib"
            "${rachis_rhi_metallib_dir}/BootstrapTriangle.ps.pnms"
            "${rachis_rhi_metallib_dir}/BootstrapCube.vs.metallib"
            "${rachis_rhi_metallib_dir}/BootstrapCube.vs.pnms"
            "${rachis_rhi_metallib_dir}/BootstrapCube.ps.metallib"
            "${rachis_rhi_metallib_dir}/BootstrapCube.ps.pnms"
            "${rachis_rhi_metallib_dir}/BootstrapCube.cs.metallib"
            "${rachis_rhi_metallib_dir}/BootstrapCube.cs.pnms"
            "${rachis_rhi_metallib_dir}/LayoutFixture.cs.metallib"
            "${rachis_rhi_metallib_dir}/LayoutFixture.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalUnusedBindings.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalUnusedBindings.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalSpecialization.7.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalSpecialization.7.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalSpecialization.13.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalSpecialization.13.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalPushConstant.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalPushConstant.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalPushConstant.set1.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalPushConstant.set1.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalPushConstant.set3.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalPushConstant.set3.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalUnusedImageBindings.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalUnusedImageBindings.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRasterInstance.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRasterInstance.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRasterInstance.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalRasterInstance.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalRasterVertex.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRasterVertex.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRasterVertex.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalRasterVertex.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.matching.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.matching.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.fewer.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.fewer.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.more.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.more.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.type.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.type.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.missing.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.missing.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.renamed.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalStageLinkage.renamed.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalComputeStorage.scatterMain.metallib"
            "${rachis_rhi_metallib_dir}/MetalComputeStorage.scatterMain.pnms"
            "${rachis_rhi_metallib_dir}/MetalComputeStorage.reduceMain.metallib"
            "${rachis_rhi_metallib_dir}/MetalComputeStorage.reduceMain.pnms"
            "${rachis_rhi_metallib_dir}/MetalShaderArithmetic.arithmetic16Main.metallib"
            "${rachis_rhi_metallib_dir}/MetalShaderArithmetic.arithmetic16Main.pnms"
            "${rachis_rhi_metallib_dir}/MetalShaderArithmetic.int64Main.metallib"
            "${rachis_rhi_metallib_dir}/MetalShaderArithmetic.int64Main.pnms"
            "${rachis_rhi_metallib_dir}/MetalShaderArithmetic.subgroupMain.metallib"
            "${rachis_rhi_metallib_dir}/MetalShaderArithmetic.subgroupMain.pnms"
            "${rachis_rhi_metallib_dir}/MetalLayeredRendering.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalLayeredRendering.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalLayeredRendering.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalLayeredRendering.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalRayQuery.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRayQuery.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.direct.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.direct.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.depth.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.depth.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.textured.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.textured.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.textured.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.textured.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalHeadlessParity.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.single.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.single.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.multi.cs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.multi.cs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.single.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.single.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.single.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.single.ps.pnms"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.multi.vs.metallib"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.multi.vs.pnms"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.multi.ps.metallib"
            "${rachis_rhi_metallib_dir}/MetalRemapContract.multi.ps.pnms"
            "${rachis_rhi_metallib_dir}/MipmapAfterRendering.vs.metallib"
            "${rachis_rhi_metallib_dir}/MipmapAfterRendering.vs.pnms"
            "${rachis_rhi_metallib_dir}/MipmapAfterRendering.slow.ps.metallib"
            "${rachis_rhi_metallib_dir}/MipmapAfterRendering.slow.ps.pnms"
            "${rachis_rhi_metallib_dir}/MipmapAfterRendering.blend.ps.metallib"
            "${rachis_rhi_metallib_dir}/MipmapAfterRendering.blend.ps.pnms"
    )
endif()

file(GLOB_RECURSE rachis_rhi_public_headers
    CONFIGURE_DEPENDS
    "${rachis_rhi_public_include_dir}/Rachis/*.h"
)

foreach(header IN LISTS rachis_rhi_public_headers)
    file(READ "${header}" contents)
    if(contents MATCHES "#[ \t]*include[^\n]*(windows\\.h|d3d12\\.h|dxgi|wrl|vulkan\\.h|volk\\.h|vk_mem_alloc|Metal/Metal\\.h)"
        OR contents MATCHES "rachis::d3d12"
        OR contents MATCHES "Vk(Instance|Device|Queue|SwapchainKHR)"
        OR contents MATCHES "CAMetalLayer"
        OR contents MATCHES "#import")
        message(FATAL_ERROR
            "Public RHI header '${header}' exposes a Windows, D3D12, Vulkan, or Metal implementation detail.")
    endif()
endforeach()

function(rachis_configure_rhi_target target)
    target_compile_features(${target} PUBLIC cxx_std_23)

    if(MSVC)
        target_compile_options(${target}
            PRIVATE
                /W4
                /permissive-
                /utf-8
        )
    else()
        target_compile_options(${target}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()
endfunction()

if(RACHIS_ENABLE_D3D12)
add_library(RachisRHID3D12 STATIC
    src/D3D12/D3D12AccelerationStructure.cpp
    src/D3D12/D3D12Binding.cpp
    src/D3D12/D3D12CommandRecording.cpp
    src/D3D12/D3D12DebugMarker.cpp
    src/D3D12/D3D12Device.cpp
    src/D3D12/D3D12Diagnostics.cpp
    src/D3D12/D3D12Descriptor.cpp
    src/D3D12/D3D12FillAndClear.cpp
    src/D3D12/D3D12Mipmap.cpp
    src/D3D12/D3D12Pipeline.cpp
    src/D3D12/D3D12Query.cpp
    src/D3D12/D3D12Queue.cpp
    src/D3D12/D3D12RayTracingPipeline.cpp
    src/D3D12/D3D12Residency.cpp
    src/D3D12/D3D12Resource.cpp
    src/D3D12/D3D12ResourceBarriers.cpp
    src/D3D12/D3D12ResourceCopies.cpp
    src/D3D12/D3D12Sampler.cpp
    src/D3D12/D3D12SparseAndFeedback.cpp
    src/D3D12/D3D12Swapchain.cpp
    src/D3D12/D3D12View.cpp
    "${rachis_generate_mipmaps_1d_header}"
    "${rachis_generate_mipmaps_header}"
    "${rachis_generate_mipmaps_3d_header}"
    "${rachis_resolve_depth_vertex_header}"
    "${rachis_resolve_depth_pixel_header}"
)
add_dependencies(RachisRHID3D12 RachisRHIShaders)

target_include_directories(RachisRHID3D12
    PUBLIC
        "$<BUILD_INTERFACE:${rachis_rhi_public_include_dir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${rachis_rhi_source_dir}"
        "${rachis_rhi_generated_dir}"
)

target_compile_definitions(RachisRHID3D12
    PRIVATE
        RACHIS_ENABLE_DEBUG_MARKERS=$<BOOL:${RACHIS_ENABLE_DEBUG_MARKERS}>
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        NOGDI
)

if(RACHIS_ENABLE_DEBUG_MARKERS)
    target_compile_definitions(RachisRHID3D12 PRIVATE USE_PIX)
endif()

target_link_libraries(RachisRHID3D12
    PRIVATE
        "$<BUILD_INTERFACE:D3D12MemoryAllocator>"
        d3d12
        dxgi
        dxguid
)

if(RACHIS_ENABLE_DEBUG_MARKERS)
    target_link_libraries(RachisRHID3D12 PRIVATE RachisWinPixEventRuntime)
endif()

rachis_configure_rhi_target(RachisRHID3D12)
add_library(Rachis::RHID3D12 ALIAS RachisRHID3D12)
endif()

if(RACHIS_ENABLE_VULKAN)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    add_library(RachisRHIVulkan STATIC
        src/Interop.cpp
        src/Vulkan/VulkanAcceleration.cpp
        src/Vulkan/VulkanBarriers.cpp
        src/Vulkan/VulkanBindings.cpp
        src/Vulkan/VulkanCommands.cpp
        src/Vulkan/VulkanPipelines.cpp
        src/Vulkan/VulkanRayTracing.cpp
        src/Vulkan/VulkanQueries.cpp
        src/Vulkan/VulkanCopies.cpp
        src/Vulkan/VulkanDevice.cpp
        src/Vulkan/VulkanDiagnostics.cpp
        src/Vulkan/VulkanFactory.cpp
        src/Vulkan/VulkanGraphics.cpp
        src/Vulkan/VulkanHeaps.cpp
        src/Vulkan/VulkanImageUtilities.cpp
        src/Vulkan/VulkanIndirectDispatch.cpp
        src/Vulkan/VulkanQueues.cpp
        src/Vulkan/VulkanResources.cpp
        src/Vulkan/VulkanRuntime.cpp
        src/Vulkan/VulkanShaders.cpp
        src/Vulkan/VulkanSwapchain.cpp
    )
    target_include_directories(RachisRHIVulkan
        PUBLIC
            "$<BUILD_INTERFACE:${rachis_rhi_public_include_dir}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
            "${rachis_rhi_source_dir}"
            "${rachis_rhi_spirv_dir}"
    )
    add_dependencies(RachisRHIVulkan RachisRHISpirvShaders)
    target_compile_definitions(RachisRHIVulkan
        PRIVATE
            RACHIS_ENABLE_D3D12=$<BOOL:${RACHIS_ENABLE_D3D12}>
            RACHIS_ENABLE_VULKAN=1
            VK_NO_PROTOTYPES
            VK_ENABLE_BETA_EXTENSIONS
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    )
    if(WIN32)
        target_compile_definitions(RachisRHIVulkan
            PRIVATE
                VK_USE_PLATFORM_WIN32_KHR
                WIN32_LEAN_AND_MEAN
                NOMINMAX
        )
    endif()
    target_link_libraries(RachisRHIVulkan
        PRIVATE
            Threads::Threads
            "$<BUILD_INTERFACE:volk>"
            "$<BUILD_INTERFACE:Vulkan-Headers>"
            "$<BUILD_INTERFACE:VulkanMemoryAllocator>"
    )
    rachis_configure_rhi_target(RachisRHIVulkan)
    add_library(Rachis::RHIVulkan ALIAS RachisRHIVulkan)
endif()

if(RACHIS_ENABLE_METAL)
    add_library(RachisRHIMetal STATIC
        src/Metal/MetalBarriers.mm
        src/Metal/MetalBindings.mm
        src/Metal/MetalBootstrap.mm
        src/Metal/MetalCommand.mm
        src/Metal/MetalCopies.mm
        src/Metal/MetalDevice.mm
        src/Metal/MetalDiagnostics.mm
        src/Metal/MetalEncoding.mm
        src/Metal/MetalFillAndClear.mm
        src/Metal/MetalIndirectRecords.mm
        src/Metal/MetalPipelines.mm
        src/Metal/MetalQueries.mm
        src/Metal/MetalResources.mm
        src/Metal/MetalShaderManifest.mm
        src/Metal/MetalSwapchain.mm
        "${rachis_rhi_metallib_dir}/RachisGenerateMipmaps2D.pnms.h"
        "${rachis_rhi_metallib_dir}/RachisGenerateMipmaps3D.pnms.h"
    )
    target_include_directories(RachisRHIMetal
        PUBLIC
            "$<BUILD_INTERFACE:${rachis_rhi_public_include_dir}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
            "${rachis_rhi_source_dir}"
            "${rachis_rhi_metallib_dir}"
    )
    if(TARGET RachisRHIMetalShaders)
        add_dependencies(RachisRHIMetal RachisRHIMetalShaders)
    endif()
    target_compile_definitions(RachisRHIMetal
        PRIVATE
            RACHIS_ENABLE_D3D12=0
            RACHIS_ENABLE_VULKAN=0
            RACHIS_ENABLE_METAL=1
    )
    target_compile_options(RachisRHIMetal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(RachisRHIMetal
        PRIVATE
            "${RACHIS_METAL_FRAMEWORK}"
            "${RACHIS_FOUNDATION_FRAMEWORK}"
            "${RACHIS_QUARTZCORE_FRAMEWORK}"
            "${RACHIS_COREGRAPHICS_FRAMEWORK}"
    )
    set_target_properties(RachisRHIMetal PROPERTIES LINKER_LANGUAGE CXX)
    rachis_configure_rhi_target(RachisRHIMetal)
    add_library(Rachis::RHIMetal ALIAS RachisRHIMetal)
endif()

add_library(RachisRHI STATIC
    src/BackendFactory.cpp
    src/Command.cpp
    src/Device.cpp
    src/PipelineLayoutDerivation.cpp
    src/RayTracingShaderTableLayout.cpp
    src/Interop.cpp
)

add_library(Rachis::RHI ALIAS RachisRHI)

set_target_properties(RachisRHI PROPERTIES
    RACHIS_ENABLE_D3D12 "${RACHIS_ENABLE_D3D12}"
    RACHIS_ENABLE_VULKAN "${RACHIS_ENABLE_VULKAN}"
    RACHIS_ENABLE_METAL "${RACHIS_ENABLE_METAL}"
    EXPORT_PROPERTIES "RACHIS_ENABLE_D3D12;RACHIS_ENABLE_VULKAN;RACHIS_ENABLE_METAL"
)
# rachis_add_shader passes this directory to the shader compilers, so shaders can include Rachis/DrawIndex.hlsli.
# It is not exported: RachisConfig.cmake sets it to the installed include directory.
set_target_properties(RachisRHI PROPERTIES RACHIS_SHADER_INCLUDE_DIR "${rachis_rhi_public_include_dir}")

target_include_directories(RachisRHI
    PUBLIC
        "$<BUILD_INTERFACE:${rachis_rhi_public_include_dir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${rachis_rhi_source_dir}"
)

if(RACHIS_ENABLE_D3D12)
    target_link_libraries(RachisRHI PRIVATE RachisRHID3D12)
endif()
if(RACHIS_ENABLE_VULKAN)
    target_link_libraries(RachisRHI PRIVATE RachisRHIVulkan)
endif()
if(RACHIS_ENABLE_METAL)
    target_link_libraries(RachisRHI PRIVATE RachisRHIMetal)
endif()
target_compile_definitions(RachisRHI
    PRIVATE
        RACHIS_ENABLE_D3D12=$<BOOL:${RACHIS_ENABLE_D3D12}>
        RACHIS_ENABLE_VULKAN=$<BOOL:${RACHIS_ENABLE_VULKAN}>
        RACHIS_ENABLE_METAL=$<BOOL:${RACHIS_ENABLE_METAL}>
    PUBLIC
        RACHIS_ENABLE_METAL=$<BOOL:${RACHIS_ENABLE_METAL}>
)

rachis_configure_rhi_target(RachisRHI)

add_library(RachisRHIShared SHARED src/C/rachis.cpp src/Interop.cpp)
add_library(Rachis::C ALIAS RachisRHIShared)
target_compile_features(RachisRHIShared PRIVATE cxx_std_23)
target_compile_definitions(RachisRHIShared PRIVATE RACHIS_C_EXPORTS)
target_include_directories(RachisRHIShared
    PUBLIC
        "$<BUILD_INTERFACE:${rachis_rhi_public_include_dir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${rachis_rhi_source_dir}")
target_link_libraries(RachisRHIShared PRIVATE Rachis::RHI)
set_target_properties(RachisRHIShared PROPERTIES
    OUTPUT_NAME rachis
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
rachis_configure_rhi_target(RachisRHIShared)

set(rachis_c_abi_header "${rachis_rhi_public_include_dir}/Rachis/rachis.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${rachis_c_abi_header}")
file(STRINGS "${rachis_c_abi_header}" rachis_c_abi_lines
    REGEX "^[ \t]*#[ \t]*define[ \t]+RACHIS_ABI_VERSION[ \t]")
list(LENGTH rachis_c_abi_lines rachis_c_abi_line_count)
if(NOT rachis_c_abi_line_count EQUAL 1
        OR NOT rachis_c_abi_lines MATCHES "RACHIS_ABI_VERSION[ \t]+([0-9]+)[uU]?([ \t]|$)")
    message(FATAL_ERROR "Could not read one numeric RACHIS_ABI_VERSION from ${rachis_c_abi_header}.")
endif()
set(rachis_c_abi_version "${CMAKE_MATCH_1}")
if(NOT RACHIS_PACKAGE_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
    message(FATAL_ERROR "RACHIS_PACKAGE_VERSION must be <major>.<minor>.<patch> (got '${RACHIS_PACKAGE_VERSION}').")
endif()
set(rachis_c_library_version "${rachis_c_abi_version}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
set_target_properties(RachisRHIShared PROPERTIES
    SOVERSION "${rachis_c_abi_version}"
    VERSION "${rachis_c_library_version}")

set_target_properties(RachisRHIShared PROPERTIES INTERFACE_COMPILE_FEATURES "")

if(APPLE)
    set(rachis_c_export_list "${CMAKE_CURRENT_SOURCE_DIR}/src/C/rachis.exported_symbols")
    target_link_options(RachisRHIShared PRIVATE "LINKER:-exported_symbols_list,${rachis_c_export_list}")
    set_property(TARGET RachisRHIShared APPEND PROPERTY LINK_DEPENDS "${rachis_c_export_list}")
elseif(UNIX)
    set(rachis_c_export_list "${CMAKE_CURRENT_SOURCE_DIR}/src/C/rachis.version")
    target_link_options(RachisRHIShared PRIVATE "LINKER:--version-script=${rachis_c_export_list}")
    set_property(TARGET RachisRHIShared APPEND PROPERTY LINK_DEPENDS "${rachis_c_export_list}")
endif()
if(RACHIS_ENABLE_D3D12 AND TARGET RachisWinPixEventRuntime)
    target_link_libraries(RachisRHIShared INTERFACE "$<LINK_ONLY:RachisWinPixEventRuntime>")
    add_custom_target(RachisWinPixEventRuntimeCopy
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:RachisRHIShared>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:RachisWinPixEventRuntime>"
            "$<TARGET_FILE_DIR:RachisRHIShared>"
        COMMENT "Copying WinPixEventRuntime.dll beside rachis.dll"
        VERBATIM)
    add_dependencies(RachisRHIShared RachisWinPixEventRuntimeCopy)
endif()

if(rachis_build_tests)
    target_compile_definitions(RachisRHI PRIVATE RACHIS_TESTING)
    if(RACHIS_ENABLE_D3D12)
        target_compile_definitions(RachisRHID3D12 PRIVATE RACHIS_TESTING)
    endif()
    if(RACHIS_ENABLE_VULKAN)
        target_compile_definitions(RachisRHIVulkan PRIVATE RACHIS_TESTING)
    endif()
    if(RACHIS_ENABLE_METAL)
        target_compile_definitions(RachisRHIMetal PRIVATE RACHIS_TESTING)
    endif()

    find_package(Catch2 3.15.3 EXACT CONFIG QUIET)
    if(NOT TARGET Catch2::Catch2WithMain)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG 8b08d4d79514f45f7e4ce2a607ac9c94e920d1bb
            EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    if(DEFINED catch2_SOURCE_DIR AND EXISTS "${catch2_SOURCE_DIR}/extras/Catch.cmake")
        include("${catch2_SOURCE_DIR}/extras/Catch.cmake")
    elseif(DEFINED Catch2_DIR AND EXISTS "${Catch2_DIR}/Catch.cmake")
        include("${Catch2_DIR}/Catch.cmake")
    else()
        message(FATAL_ERROR "Catch2's CTest integration module was not found.")
    endif()

    set(rachis_c_abi_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/CAbiShaders")
    set(rachis_c_abi_shader_source "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/CAbiComputeWrite.hlsl")
    set(rachis_c_abi_push_source "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/CAbiComputePush.hlsl")
    set(rachis_c_abi_graphics_source "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/CAbiGraphicsTriangle.hlsl")
    set(rachis_c_abi_vulkan_graphics_source "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/CAbiGraphicsTriangleVulkan.hlsl")
    set(rachis_c_abi_indexed_source "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/CAbiGraphicsIndexed.hlsl")
    set(rachis_c_abi_shader_outputs)
    if(RACHIS_ENABLE_D3D12)
        rachis_cook_dxc_dxil(
            SOURCE "${rachis_c_abi_shader_source}"
            ENTRY main
            PROFILE cs_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiComputeWrite.dxil"
            COMMENT "Compiling the Rachis C ABI compute contract shader"
        )
        list(APPEND rachis_c_abi_shader_outputs "${rachis_c_abi_shader_dir}/CAbiComputeWrite.dxil")
        rachis_cook_dxc_dxil(
            SOURCE "${rachis_c_abi_push_source}"
            ENTRY main
            PROFILE cs_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiComputePush.dxil"
            COMMENT "Compiling the Rachis C ABI compute push-constant shader"
        )
        list(APPEND rachis_c_abi_shader_outputs "${rachis_c_abi_shader_dir}/CAbiComputePush.dxil")
        foreach(stage IN ITEMS vertex pixel)
            if(stage STREQUAL "vertex")
                set(profile vs_6_0)
                set(entry vertexMain)
            else()
                set(profile ps_6_0)
                set(entry pixelMain)
            endif()
            rachis_cook_dxc_dxil(
                SOURCE "${rachis_c_abi_graphics_source}"
                ENTRY ${entry}
                PROFILE ${profile}
                OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.${stage}.dxil"
                COMMENT "Compiling the Rachis C ABI graphics ${stage} shader"
            )
            list(APPEND rachis_c_abi_shader_outputs "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.${stage}.dxil")
            rachis_cook_dxc_dxil(
                SOURCE "${rachis_c_abi_indexed_source}"
                ENTRY ${entry}
                PROFILE ${profile}
                OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.${stage}.dxil"
                COMMENT "Compiling the Rachis C ABI indexed graphics ${stage} shader"
            )
            list(APPEND rachis_c_abi_shader_outputs "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.${stage}.dxil")
        endforeach()
    endif()
    if(RACHIS_ENABLE_VULKAN)
        rachis_cook_dxc_spirv(
            SOURCE "${rachis_c_abi_shader_source}"
            ENTRY main
            PROFILE cs_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiComputeWrite.spv"
            COMMENT "Cooking Vulkan SPIR-V CAbiComputeWrite"
        )
        list(APPEND rachis_c_abi_shader_outputs "${rachis_c_abi_shader_dir}/CAbiComputeWrite.spv")
        rachis_cook_dxc_spirv(
            SOURCE "${rachis_c_abi_push_source}"
            ENTRY main
            PROFILE cs_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiComputePush.spv"
            COMMENT "Cooking Vulkan SPIR-V CAbiComputePush"
        )
        list(APPEND rachis_c_abi_shader_outputs "${rachis_c_abi_shader_dir}/CAbiComputePush.spv")
        rachis_cook_dxc_spirv(
            SOURCE "${rachis_c_abi_vulkan_graphics_source}"
            ENTRY vertexMain
            PROFILE vs_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.vertex.spv"
            COMMENT "Cooking Vulkan SPIR-V CAbiGraphicsTriangle vertex"
        )
        rachis_cook_dxc_spirv(
            SOURCE "${rachis_c_abi_vulkan_graphics_source}"
            ENTRY pixelMain
            PROFILE ps_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.pixel.spv"
            COMMENT "Cooking Vulkan SPIR-V CAbiGraphicsTriangle pixel"
        )
        rachis_cook_dxc_spirv(
            SOURCE "${rachis_c_abi_indexed_source}"
            ENTRY vertexMain
            PROFILE vs_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.vertex.spv"
            COMMENT "Cooking Vulkan SPIR-V CAbiGraphicsIndexed vertex"
        )
        rachis_cook_dxc_spirv(
            SOURCE "${rachis_c_abi_indexed_source}"
            ENTRY pixelMain
            PROFILE ps_6_0
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.pixel.spv"
            COMMENT "Cooking Vulkan SPIR-V CAbiGraphicsIndexed pixel"
        )
        list(APPEND rachis_c_abi_shader_outputs
            "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.vertex.spv"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.pixel.spv"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.vertex.spv"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.pixel.spv")
    endif()
    if(RACHIS_ENABLE_METAL)
        rachis_cook_slang_metallib(
            SOURCE "${rachis_c_abi_shader_source}"
            ENTRY main
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiComputeWrite.metallib"
            NO_WARNINGS_AS_ERRORS
            COMMENT "Cooking Metal PNMS CAbiComputeWrite"
        )
        rachis_cook_slang_metallib(
            SOURCE "${rachis_c_abi_push_source}"
            ENTRY main
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiComputePush.metallib"
            NO_WARNINGS_AS_ERRORS
            COMMENT "Cooking Metal PNMS CAbiComputePush"
        )
        rachis_cook_slang_metallib(
            SOURCE "${rachis_c_abi_graphics_source}"
            ENTRY vertexMain
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.vertex.metallib"
            COMMENT "Cooking Metal PNMS CAbiGraphicsTriangle vertex"
        )
        rachis_cook_slang_metallib(
            SOURCE "${rachis_c_abi_graphics_source}"
            ENTRY pixelMain
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.pixel.metallib"
            COMMENT "Cooking Metal PNMS CAbiGraphicsTriangle pixel"
        )
        rachis_cook_slang_metallib(
            SOURCE "${rachis_c_abi_indexed_source}"
            ENTRY vertexMain
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.vertex.metallib"
            COMMENT "Cooking Metal PNMS CAbiGraphicsIndexed vertex"
        )
        rachis_cook_slang_metallib(
            SOURCE "${rachis_c_abi_indexed_source}"
            ENTRY pixelMain
            OUTPUT "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.pixel.metallib"
            COMMENT "Cooking Metal PNMS CAbiGraphicsIndexed pixel"
        )
        list(APPEND rachis_c_abi_shader_outputs
            "${rachis_c_abi_shader_dir}/CAbiComputeWrite.pnms"
            "${rachis_c_abi_shader_dir}/CAbiComputePush.pnms"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.vertex.pnms"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.pixel.pnms"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.vertex.pnms"
            "${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.pixel.pnms")
    endif()
    if(rachis_c_abi_shader_outputs)
        add_custom_target(RachisCAbiShaders DEPENDS ${rachis_c_abi_shader_outputs})
    endif()

    function(rachis_add_c_abi_smoke target backend adapter size)
        add_executable(${target} tests/c_abi/CAbiSmoke.c)
        target_compile_features(${target} PRIVATE c_std_11)
        target_link_libraries(${target} PRIVATE Rachis::C)
        target_compile_definitions(${target} PRIVATE
            RACHIS_SMOKE_BACKEND=${backend} RACHIS_SMOKE_ADAPTER=${adapter})
        set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
        add_test(NAME ${target} COMMAND ${target})
        set(labels "size:${size}")
        if(backend STREQUAL "RACHIS_BACKEND_VULKAN")
            list(APPEND labels "requires:HardwareVulkan")
        endif()
        set_tests_properties(${target} PROPERTIES LABELS "${labels}" TIMEOUT 30)
    endfunction()

    function(rachis_add_c_abi_tests target backend adapter size shader)
        add_executable(${target} tests/c_abi/CAbiContractTests.cpp)
        if(WIN32)
            target_sources(${target} PRIVATE tests/c_abi/CAbiPresentationTests.cpp)
            target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
        endif()
        target_link_libraries(${target} PRIVATE Rachis::C Catch2::Catch2WithMain)
        rachis_configure_rhi_target(${target})

        if(backend STREQUAL "RACHIS_BACKEND_VULKAN")
            set(requires "[requires:HardwareVulkan]")
        else()
            set(requires "")
        endif()
        target_compile_definitions(${target} PRIVATE
            RACHIS_SMOKE_BACKEND=${backend}
            RACHIS_SMOKE_ADAPTER=${adapter}
            RACHIS_CABI_DEVICE_SIZE="[size:${size}]"
            RACHIS_CABI_DEVICE_REQUIRES="${requires}"
            RACHIS_CABI_SHARED_LIBRARY_PATH="$<TARGET_FILE:RachisRHIShared>"
            RACHIS_CABI_MANIFEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/C/rachis_abi_exports.txt"
            RACHIS_CABI_HEADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/include/Rachis/rachis.h")
        if(shader)
            add_dependencies(${target} RachisCAbiShaders)
            target_compile_definitions(${target} PRIVATE RACHIS_CABI_COMPUTE_SHADER_PATH="${shader}")
            if(backend STREQUAL "RACHIS_BACKEND_D3D12")
                set(graphics_extension dxil)
            elseif(backend STREQUAL "RACHIS_BACKEND_METAL")
                set(graphics_extension pnms)
            else()
                set(graphics_extension spv)
            endif()
            target_compile_definitions(${target} PRIVATE
                RACHIS_CABI_COMPUTE_PUSH_SHADER_PATH="${rachis_c_abi_shader_dir}/CAbiComputePush.${graphics_extension}"
                RACHIS_CABI_GRAPHICS_VERTEX_PATH="${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.vertex.${graphics_extension}"
                RACHIS_CABI_GRAPHICS_PIXEL_PATH="${rachis_c_abi_shader_dir}/CAbiGraphicsTriangle.pixel.${graphics_extension}"
                RACHIS_CABI_INDEXED_VERTEX_PATH="${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.vertex.${graphics_extension}"
                RACHIS_CABI_INDEXED_PIXEL_PATH="${rachis_c_abi_shader_dir}/CAbiGraphicsIndexed.pixel.${graphics_extension}")
        endif()
        catch_discover_tests(${target} TEST_PREFIX "${target}."
            DISCOVERY_MODE PRE_TEST ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 60)
    endfunction()

    if(RACHIS_ENABLE_D3D12)
        rachis_add_c_abi_smoke(RachisCAbiSmoke RACHIS_BACKEND_D3D12 RACHIS_ADAPTER_WARP medium)
        rachis_add_c_abi_tests(RachisCAbiTests RACHIS_BACKEND_D3D12 RACHIS_ADAPTER_WARP medium
            "${rachis_c_abi_shader_dir}/CAbiComputeWrite.dxil")

        target_compile_definitions(RachisCAbiTests PRIVATE RACHIS_CABI_D3D12_DEVICE_REMOVAL=1)
        target_link_libraries(RachisCAbiTests PRIVATE d3d12 dxgi)
        if(RACHIS_ENABLE_VULKAN)
            rachis_add_c_abi_smoke(RachisCAbiVulkanSmoke RACHIS_BACKEND_VULKAN RACHIS_ADAPTER_HIGH_PERFORMANCE medium)
            rachis_add_c_abi_tests(RachisCAbiVulkanTests RACHIS_BACKEND_VULKAN RACHIS_ADAPTER_HIGH_PERFORMANCE medium
                "${rachis_c_abi_shader_dir}/CAbiComputeWrite.spv")
        endif()
    elseif(RACHIS_ENABLE_METAL)
        rachis_add_c_abi_smoke(RachisCAbiSmoke RACHIS_BACKEND_METAL RACHIS_ADAPTER_HIGH_PERFORMANCE large)
        rachis_add_c_abi_tests(RachisCAbiTests RACHIS_BACKEND_METAL RACHIS_ADAPTER_HIGH_PERFORMANCE large
            "${rachis_c_abi_shader_dir}/CAbiComputeWrite.pnms")
    else()
        rachis_add_c_abi_smoke(RachisCAbiSmoke RACHIS_BACKEND_VULKAN RACHIS_ADAPTER_HIGH_PERFORMANCE large)
        rachis_add_c_abi_tests(RachisCAbiTests RACHIS_BACKEND_VULKAN RACHIS_ADAPTER_HIGH_PERFORMANCE large
            "${rachis_c_abi_shader_dir}/CAbiComputeWrite.spv")
    endif()
    if(UNIX)
        add_test(NAME RachisCAbiExports
            COMMAND "${CMAKE_COMMAND}" "-DLIBRARY=$<TARGET_FILE:RachisRHIShared>" "-DNM=${CMAKE_NM}"
                "-DAPPLE=${APPLE}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/c_abi/CheckCAbiExports.cmake")
        set_tests_properties(RachisCAbiExports PROPERTIES LABELS "size:medium" TIMEOUT 30)
    endif()

    add_library(RachisRHITestSupport INTERFACE)
    target_include_directories(RachisRHITestSupport INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/include/Rachis/TestSupport")

    add_library(RachisRHITestMain OBJECT tests/support/CatchForceVulkanMain.cpp)
    target_link_libraries(RachisRHITestMain PRIVATE RachisRHITestSupport Catch2::Catch2)
    rachis_configure_rhi_target(RachisRHITestMain)

    add_library(RachisRHITestHooks STATIC tests/support/PresentInjection.cpp)
    target_include_directories(RachisRHITestHooks PRIVATE "${rachis_rhi_source_dir}")
    target_compile_definitions(RachisRHITestHooks PRIVATE
        RACHIS_TESTING
        RACHIS_ENABLE_D3D12=$<BOOL:${RACHIS_ENABLE_D3D12}>
        RACHIS_ENABLE_VULKAN=$<BOOL:${RACHIS_ENABLE_VULKAN}>
        RACHIS_ENABLE_METAL=$<BOOL:${RACHIS_ENABLE_METAL}>)
    target_link_libraries(RachisRHITestHooks PUBLIC RachisRHITestSupport PRIVATE Rachis::RHI)
    rachis_configure_rhi_target(RachisRHITestHooks)

    if(APPLE AND RACHIS_ENABLE_METAL)
        add_library(RachisRHIMetalSurfaceTestHost OBJECT tests/support/MetalSurfaceTestHost.mm)
        target_link_libraries(RachisRHIMetalSurfaceTestHost PUBLIC RachisRHITestSupport)
        target_compile_options(RachisRHIMetalSurfaceTestHost PRIVATE -fobjc-arc)
        rachis_configure_rhi_target(RachisRHIMetalSurfaceTestHost)
    endif()

    unset(rachis_dumpbin)
    unset(rachis_readobj)
    if(MSVC AND CMAKE_LINKER)
        get_filename_component(rachis_linker_dir "${CMAKE_LINKER}" DIRECTORY)
        find_program(rachis_dumpbin dumpbin HINTS "${rachis_linker_dir}" NO_DEFAULT_PATH NO_CACHE)
        find_program(rachis_readobj llvm-readobj HINTS "${rachis_linker_dir}" NO_DEFAULT_PATH NO_CACHE)
    endif()
    set(rachis_c_abi_export_tools
        "-DNM=${CMAKE_NM}"
        "-DDUMPBIN=${rachis_dumpbin}"
        "-DREADOBJ=${rachis_readobj}")
    add_test(NAME RachisCAbiManifest
        COMMAND "${CMAKE_COMMAND}"
            "-DHEADER=${CMAKE_CURRENT_SOURCE_DIR}/include/Rachis/rachis.h"
            "-DMANIFEST=${CMAKE_CURRENT_SOURCE_DIR}/src/C/rachis_abi_exports.txt"
            "-DLIBRARY=$<TARGET_FILE:RachisRHIShared>"
            ${rachis_c_abi_export_tools}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckCAbiManifest.cmake")
    set_tests_properties(RachisCAbiManifest PROPERTIES LABELS "size:small" TIMEOUT 30)

    add_library(RachisCAbiExportLeak SHARED tests/c_abi/exportleak/ExportLeak.c)
    set_target_properties(RachisCAbiExportLeak PROPERTIES C_VISIBILITY_PRESET hidden)
    add_test(NAME RachisCAbiManifestFindsLeaks
        COMMAND "${CMAKE_COMMAND}"
            "-DHEADER=${CMAKE_CURRENT_SOURCE_DIR}/tests/c_abi/exportleak/ExportLeak.h"
            "-DMANIFEST=${CMAKE_CURRENT_SOURCE_DIR}/tests/c_abi/exportleak/ExportLeak.txt"
            "-DLIBRARY=$<TARGET_FILE:RachisCAbiExportLeak>"
            ${rachis_c_abi_export_tools}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckCAbiManifest.cmake")
    set_tests_properties(RachisCAbiManifestFindsLeaks PROPERTIES
        LABELS "size:small"
        TIMEOUT 30
        PASS_REGULAR_EXPRESSION
            "Unexpected exports: fixture_leaked_function rachis_fixture_leaked_constant( rachis_fixture_leaked_weak)?\r?\n +Missing exports: none")

    set(rachis_c_abi_freeze_fixtures "${CMAKE_CURRENT_SOURCE_DIR}/tests/c_abi/freeze")
    foreach(rachis_freeze_case IN ITEMS Reformatted:Accepts Bumped:Accepts
            Changed:Rejects JoinedDirective:Rejects Lowered:Rejects)
        string(REPLACE ":" ";" rachis_freeze_case "${rachis_freeze_case}")
        list(GET rachis_freeze_case 0 rachis_freeze_fixture)
        list(GET rachis_freeze_case 1 rachis_freeze_outcome)
        set(rachis_freeze_test RachisCAbiFreeze${rachis_freeze_outcome}${rachis_freeze_fixture})
        add_test(NAME ${rachis_freeze_test}
            COMMAND "${CMAKE_COMMAND}"
                "-DHEADER=${rachis_c_abi_freeze_fixtures}/${rachis_freeze_fixture}.txt"
                "-DBASELINE_HEADER=${rachis_c_abi_freeze_fixtures}/Baseline.txt"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckCAbiFreeze.cmake")
        set_tests_properties(${rachis_freeze_test} PROPERTIES LABELS "size:small" TIMEOUT 30)
        if(rachis_freeze_fixture STREQUAL "Lowered")
            set_tests_properties(${rachis_freeze_test} PROPERTIES
                PASS_REGULAR_EXPRESSION "RACHIS_ABI_VERSION went down from 2 in Baseline.txt to 1")
        elseif(rachis_freeze_outcome STREQUAL "Rejects")
            set_tests_properties(${rachis_freeze_test} PROPERTIES
                PASS_REGULAR_EXPRESSION "rachis.h changed since Baseline.txt but RACHIS_ABI_VERSION is still 2")
        endif()
    endforeach()
    unset(rachis_c_abi_freeze_fixtures)
    unset(rachis_freeze_case)
    unset(rachis_freeze_fixture)
    unset(rachis_freeze_outcome)
    unset(rachis_freeze_test)

    add_test(NAME RachisDeviceCreationGuards
        COMMAND "${CMAKE_COMMAND}" "-DTEST_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/tests"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/CheckDeviceCreationGuards.cmake")
    set_tests_properties(RachisDeviceCreationGuards PROPERTIES LABELS "size:small" TIMEOUT 30)

    function(rachis_add_c_abi_fault_tests target test_prefix backend adapter size skip_host_only)
        if(backend STREQUAL "RACHIS_BACKEND_D3D12")
            set(fault_backend_def RACHIS_CABI_FAULT_BACKEND_D3D12=1)
            set(fault_requires "")
            set(fault_shader_extension dxil)
        elseif(backend STREQUAL "RACHIS_BACKEND_METAL")
            set(fault_backend_def RACHIS_CABI_FAULT_BACKEND_METAL=1)
            set(fault_requires "")
            set(fault_shader_extension pnms)
        else()
            set(fault_backend_def RACHIS_CABI_FAULT_BACKEND_VULKAN=1)
            set(fault_requires "[requires:HardwareVulkan]")
            set(fault_shader_extension spv)
        endif()

        add_executable(${target}
            tests/c_abi/CAbiFaultInjectionTests.cpp
            src/C/rachis.cpp)
        target_include_directories(${target} PRIVATE "${rachis_rhi_source_dir}")
        target_compile_definitions(${target} PRIVATE
            RACHIS_TESTING
            RACHIS_C_EXPORTS
            RACHIS_ENABLE_D3D12=$<BOOL:${RACHIS_ENABLE_D3D12}>
            RACHIS_ENABLE_VULKAN=$<BOOL:${RACHIS_ENABLE_VULKAN}>
            RACHIS_ENABLE_METAL=$<BOOL:${RACHIS_ENABLE_METAL}>
            ${fault_backend_def}
            RACHIS_CABI_FAULT_SKIP_HOST_ONLY=${skip_host_only}
            RACHIS_SMOKE_BACKEND=${backend}
            RACHIS_SMOKE_ADAPTER=${adapter}
            RACHIS_CABI_DEVICE_SIZE="[size:${size}]"
            RACHIS_CABI_DEVICE_REQUIRES="${fault_requires}")
        if(WIN32)
            target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
        endif()
        if(TARGET RachisCAbiShaders)
            add_dependencies(${target} RachisCAbiShaders)
            target_compile_definitions(${target} PRIVATE
                RACHIS_CABI_COMPUTE_SHADER_PATH="${rachis_c_abi_shader_dir}/CAbiComputeWrite.${fault_shader_extension}")
        endif()
        target_link_libraries(${target} PRIVATE
            Rachis::RHI
            RachisRHITestHooks
            Catch2::Catch2WithMain)
        if(backend STREQUAL "RACHIS_BACKEND_VULKAN")
            target_link_libraries(${target} PRIVATE volk Vulkan-Headers VulkanMemoryAllocator)
            target_compile_definitions(${target} PRIVATE VK_NO_PROTOTYPES VK_ENABLE_BETA_EXTENSIONS)
        endif()
        if(backend STREQUAL "RACHIS_BACKEND_D3D12")
            target_link_libraries(${target} PRIVATE d3d12 dxgi)
        endif()
        if(APPLE AND backend STREQUAL "RACHIS_BACKEND_METAL")
            target_link_libraries(${target} PRIVATE
                RachisRHIMetalSurfaceTestHost
                "${RACHIS_APPKIT_FRAMEWORK}"
                "${RACHIS_QUARTZCORE_FRAMEWORK}"
                "${RACHIS_METAL_FRAMEWORK}")
        endif()
        if(TARGET RachisWinPixEventRuntimeCopy)
            add_dependencies(${target} RachisWinPixEventRuntimeCopy)
        endif()
        rachis_configure_rhi_target(${target})
        catch_discover_tests(${target}
            TEST_PREFIX "${test_prefix}"
            DISCOVERY_MODE PRE_TEST
            ADD_TAGS_AS_LABELS
            PROPERTIES TIMEOUT 60)
    endfunction()

    if(RACHIS_ENABLE_D3D12)
        rachis_add_c_abi_fault_tests(
            RachisCAbiFaultInjectionTests "RachisCAbiFault." RACHIS_BACKEND_D3D12 RACHIS_ADAPTER_WARP medium 0)
        if(RACHIS_ENABLE_VULKAN)
            rachis_add_c_abi_fault_tests(
                RachisCAbiVulkanFaultInjectionTests "RachisCAbiVulkanFault."
                RACHIS_BACKEND_VULKAN RACHIS_ADAPTER_HIGH_PERFORMANCE large 1)
        endif()
    elseif(RACHIS_ENABLE_METAL)
        rachis_add_c_abi_fault_tests(
            RachisCAbiFaultInjectionTests "RachisCAbiFault." RACHIS_BACKEND_METAL RACHIS_ADAPTER_HIGH_PERFORMANCE large 0)
    else()
        rachis_add_c_abi_fault_tests(
            RachisCAbiFaultInjectionTests "RachisCAbiFault." RACHIS_BACKEND_VULKAN RACHIS_ADAPTER_HIGH_PERFORMANCE large 0)
    endif()

    if(UNIX AND NOT APPLE AND RACHIS_ENABLE_VULKAN)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(RACHIS_LINUX_WSI REQUIRED IMPORTED_TARGET wayland-client xcb)
        pkg_get_variable(rachis_wayland_protocols_dir wayland-protocols pkgdatadir)
        find_program(RACHIS_WAYLAND_SCANNER wayland-scanner)
        if(NOT rachis_wayland_protocols_dir OR NOT RACHIS_WAYLAND_SCANNER)
            message(FATAL_ERROR
                "The Linux Vulkan WSI tests need wayland-protocols and wayland-scanner; "
                "see docs/building.md.")
        endif()
        set(rachis_xdg_shell_xml "${rachis_wayland_protocols_dir}/stable/xdg-shell/xdg-shell.xml")
        set(rachis_linux_wsi_dir "${CMAKE_CURRENT_BINARY_DIR}/linux-wsi")
        add_custom_command(
            OUTPUT
                "${rachis_linux_wsi_dir}/xdg-shell-client-protocol.h"
                "${rachis_linux_wsi_dir}/xdg-shell-protocol.c"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${rachis_linux_wsi_dir}"
            COMMAND "${RACHIS_WAYLAND_SCANNER}" client-header
                "${rachis_xdg_shell_xml}" "${rachis_linux_wsi_dir}/xdg-shell-client-protocol.h"
            COMMAND "${RACHIS_WAYLAND_SCANNER}" private-code
                "${rachis_xdg_shell_xml}" "${rachis_linux_wsi_dir}/xdg-shell-protocol.c"
            DEPENDS "${rachis_xdg_shell_xml}"
            COMMENT "Generating the xdg-shell client protocol for the Linux WSI tests"
            VERBATIM
        )
        add_custom_target(RachisRHILinuxWaylandProtocols
            DEPENDS
                "${rachis_linux_wsi_dir}/xdg-shell-client-protocol.h"
                "${rachis_linux_wsi_dir}/xdg-shell-protocol.c"
        )

        add_library(RachisRHILinuxWaylandProtocol STATIC "${rachis_linux_wsi_dir}/xdg-shell-protocol.c")
        add_dependencies(RachisRHILinuxWaylandProtocol RachisRHILinuxWaylandProtocols)
        target_link_libraries(RachisRHILinuxWaylandProtocol PRIVATE PkgConfig::RACHIS_LINUX_WSI)
        add_library(RachisRHILinuxSurfaceTestHost OBJECT tests/support/LinuxSurfaceTestHost.cpp)
        add_dependencies(RachisRHILinuxSurfaceTestHost RachisRHILinuxWaylandProtocols)
        target_include_directories(RachisRHILinuxSurfaceTestHost PRIVATE "${rachis_linux_wsi_dir}")
        target_link_libraries(RachisRHILinuxSurfaceTestHost
            PUBLIC RachisRHITestSupport RachisRHILinuxWaylandProtocol PkgConfig::RACHIS_LINUX_WSI)
        rachis_configure_rhi_target(RachisRHILinuxSurfaceTestHost)
    endif()

    if(RACHIS_ENABLE_D3D12)
    set(rachis_rhi_dxil_dir "${CMAKE_CURRENT_BINARY_DIR}/Generated/dxil")
    set(rachis_rhi_test_dxil_outputs)
    set(rachis_rhi_test_dxil_sources)
    function(rachis_rhi_test_dxil name source profile)
        cmake_parse_arguments(DXIL "" "ENTRY" "OPTIONS;INCLUDE_DIRECTORIES" ${ARGN})
        set(output "${rachis_rhi_dxil_dir}/${name}")
        rachis_cook_dxc_dxil(
            SOURCE "${source}"
            ENTRY "${DXIL_ENTRY}"
            PROFILE "${profile}"
            OUTPUT "${output}"
            OPTIONS ${DXIL_OPTIONS}
            INCLUDE_DIRECTORIES ${DXIL_INCLUDE_DIRECTORIES}
            COMMENT "Compiling DXIL ${name}"
        )
        set(rachis_rhi_test_dxil_outputs ${rachis_rhi_test_dxil_outputs} "${output}" PARENT_SCOPE)
        set(rachis_rhi_test_dxil_sources ${rachis_rhi_test_dxil_sources} "${source}" PARENT_SCOPE)
    endfunction()
    add_executable(RachisRHITests
        tests/core/FrontendDefaultsAndInternalsContractTests.cpp
        tests/core/FrontendValidationContractTests.cpp
        tests/core/HandleTableTests.cpp
        tests/core/PipelineLayoutDerivationTests.cpp
        tests/core/RayTracingShaderTableLayoutTests.cpp
        tests/core/ShaderBinaryToolContractTests.cpp
        tests/core/ShaderInterfaceReflectionTests.cpp
        tests/core/StageLinkageTests.cpp
        tests/d3d12/binding/BindingBatchContractTests.cpp
        tests/d3d12/binding/BindingContractTests.cpp
        tests/d3d12/binding/BindingSetUpdateContractTests.cpp
        tests/d3d12/binding/DynamicBindingContractTests.cpp
        tests/d3d12/binding/PartiallyBoundBindingContractTests.cpp
        tests/d3d12/binding/UpdateAfterBindBindingContractTests.cpp
        tests/d3d12/binding/VariableBindingContractTests.cpp
        tests/d3d12/commands/BufferBarrierRangeContractTests.cpp
        tests/d3d12/commands/CommandPoolContractTests.cpp
        tests/d3d12/commands/QueueContractTests.cpp
        tests/d3d12/commands/ResourceBarrierBatchContractTests.cpp
        tests/d3d12/commands/TextureLayoutContractTests.cpp
        tests/d3d12/commands/TimelineObservationContractTests.cpp
        tests/d3d12/compute/ComputeContractTests.cpp
        tests/d3d12/compute/IndirectDispatchBatchContractTests.cpp
        tests/d3d12/device/DeviceContractTests.cpp
        tests/d3d12/device/HandleProvenanceContractTests.cpp
        tests/d3d12/diagnostics/DebugMarkerContractTests.cpp
        tests/d3d12/diagnostics/DiagnosticsContractTests.cpp
        tests/d3d12/internals/D3D12AccelerationStructureLeaseTests.cpp
        tests/d3d12/internals/D3D12BarrierTranslationTests.cpp
        tests/d3d12/internals/D3D12DxilSignatureTests.cpp
        tests/d3d12/internals/D3D12LegacyTextureStateTests.cpp
        tests/d3d12/internals/D3D12RuntimePoliciesTests.cpp
        tests/d3d12/internals/D3D12SwapchainLeaseTests.cpp
        tests/d3d12/interop/SharedBufferInteropContractTests.cpp
        tests/d3d12/interop/SharedTextureInteropContractTests.cpp
        tests/d3d12/memory/ManualResidencyContractTests.cpp
        tests/d3d12/memory/PlacedBufferHeapContractTests.cpp
        tests/d3d12/memory/PlacedTextureHeapContractTests.cpp
        tests/d3d12/memory/ResidencyPriorityContractTests.cpp
        tests/d3d12/memory/ResourceAliasingContractTests.cpp
        tests/d3d12/memory/ResourceHeapResidencyContractTests.cpp
        tests/d3d12/memory/ResourceMemoryRequirementsContractTests.cpp
        tests/d3d12/memory/ResourceMemorySnapshotContractTests.cpp
        tests/d3d12/memory/SparseBufferContractTests.cpp
        tests/d3d12/memory/SparseBufferShaderContractTests.cpp
        tests/d3d12/memory/SparseTextureContractTests.cpp
        tests/d3d12/memory/SparseTextureShaderContractTests.cpp
        tests/d3d12/pipelines/PipelineCacheContractTests.cpp
        tests/d3d12/pipelines/PipelineCompilationContractTests.cpp
        tests/d3d12/pipelines/PipelineContractTests.cpp
        tests/d3d12/pipelines/StageLinkageContractTests.cpp
        tests/d3d12/presentation/D3D12PresentationProgressContractTests.cpp
        tests/d3d12/presentation/PresentationContractTests.cpp
        tests/d3d12/queries/ConditionalExecutionContractTests.cpp
        tests/d3d12/queries/GpuProfileScopeContractTests.cpp
        tests/d3d12/queries/OcclusionQueryContractTests.cpp
        tests/d3d12/queries/PipelineStatisticsQueryContractTests.cpp
        tests/d3d12/queries/QueueClockCalibrationContractTests.cpp
        tests/d3d12/queries/TimestampQueryContractTests.cpp
        tests/d3d12/raytracing/RayQueryContractTests.cpp
        tests/d3d12/raytracing/TopLevelInstanceBufferContractTests.cpp
        tests/d3d12/raytracing/RayTracingPipelineContractTests.cpp
        tests/d3d12/rendering/AttachmentClearContractTests.cpp
        tests/d3d12/rendering/DepthBoundsContractTests.cpp
        tests/d3d12/rendering/DualSourceBlendContractTests.cpp
        tests/d3d12/rendering/DynamicDepthBiasContractTests.cpp
        tests/d3d12/rendering/FragmentShaderBuiltinsContractTests.cpp
        tests/d3d12/rendering/GeometryShaderContractTests.cpp
        tests/d3d12/rendering/GraphicsContractTests.cpp
        tests/d3d12/rendering/IndependentStencilReferenceContractTests.cpp
        tests/d3d12/rendering/DrawIndexContractTests.cpp
        tests/d3d12/rendering/IndirectDrawContractTests.cpp
        tests/d3d12/rendering/LayeredRenderingContractTests.cpp
        tests/d3d12/rendering/LogicOperationContractTests.cpp
        tests/d3d12/rendering/MeshShaderContractTests.cpp
        tests/d3d12/rendering/MultiViewportContractTests.cpp
        tests/d3d12/rendering/MultisampleResolveContractTests.cpp
        tests/d3d12/rendering/MultiviewRenderingContractTests.cpp
        tests/d3d12/rendering/PackedDepthStencilResolveContractTests.cpp
        tests/d3d12/rendering/PrimitiveRestartContractTests.cpp
        tests/d3d12/rendering/RasterizationContractTests.cpp
        tests/d3d12/rendering/RasterizerOrderedStorageContractTests.cpp
        tests/d3d12/rendering/StreamOutputContractTests.cpp
        tests/d3d12/rendering/TessellationShaderContractTests.cpp
        tests/d3d12/rendering/TriangleFanContractTests.cpp
        tests/d3d12/rendering/VariableRateShadingContractTests.cpp
        tests/d3d12/rendering/VertexInputContractTests.cpp
        tests/d3d12/resources/BlockCompressedFormatContractTests.cpp
        tests/d3d12/resources/BufferContractTests.cpp
        tests/d3d12/resources/BufferCopyBatchContractTests.cpp
        tests/d3d12/resources/FillAndClearContractTests.cpp
        tests/d3d12/resources/MinMipSamplerFeedbackContractTests.cpp
        tests/d3d12/resources/MipRegionUsedSamplerFeedbackContractTests.cpp
        tests/d3d12/resources/MipmapContractTests.cpp
        tests/d3d12/resources/MipmapDimensionContractTests.cpp
        tests/d3d12/resources/MipmapShaderResourceContractTests.cpp
        tests/d3d12/resources/MipmapSrgbContractTests.cpp
        tests/d3d12/resources/PackedDepthStencilCopyContractTests.cpp
        tests/d3d12/resources/SamplerContractTests.cpp
        tests/d3d12/resources/SamplerReductionContractTests.cpp
        tests/d3d12/resources/TextureContractTests.cpp
        tests/d3d12/resources/TextureCopyBatchContractTests.cpp
        tests/d3d12/resources/TextureTransferLayoutContractTests.cpp
        tests/d3d12/resources/TextureViewComponentSwizzleContractTests.cpp
        tests/d3d12/resources/TypedIntegerFormatContractTests.cpp
        tests/d3d12/shaders/Shader16BitContractTests.cpp
        tests/d3d12/shaders/ShaderDistanceContractTests.cpp
        tests/d3d12/shaders/ShaderInt64AtomicContractTests.cpp
        tests/d3d12/shaders/ShaderInt64ContractTests.cpp
        tests/d3d12/shaders/ShaderInterfaceContractTests.cpp
        tests/d3d12/shaders/ShaderInterfaceSnapshotContractTests.cpp
        tests/d3d12/shaders/ShaderLibraryContractTests.cpp
        tests/d3d12/shaders/ShaderSpecializationContractTests.cpp
        tests/d3d12/shaders/SubgroupOperationsContractTests.cpp
        tests/metal/commands/MetalBarrierContractTests.cpp
        tests/metal/commands/MetalCommandPoolContractTests.cpp
        tests/metal/commands/MetalQueueContractTests.cpp
        tests/metal/device/MetalBackendDispatchContractTests.cpp
        tests/metal/device/MetalDeviceLossContractTests.cpp
        tests/metal/device/MetalHandleProvenanceContractTests.cpp
        tests/metal/device/MetalHeadlessDeviceContractTests.cpp
        tests/metal/device/MetalLifecycleContractTests.cpp
        tests/metal/device/MetalMacosProfileContractTests.cpp
        tests/metal/diagnostics/MetalDiagnosticsContractTests.cpp
        tests/metal/memory/MetalBudgetContractTests.cpp
        tests/metal/resources/MetalBufferContractTests.cpp
        tests/metal/resources/MetalCopiesContractTests.cpp
        tests/metal/resources/MetalTextureContractTests.cpp
        tests/portable/PortableBackendContractTests.cpp
        tests/portable/GeneratedShaderInterfaceTests.cpp
        tests/vulkan/binding/VulkanBindingContractTests.cpp
        tests/vulkan/binding/VulkanDescriptorIndexingContractTests.cpp
        tests/vulkan/binding/VulkanUpdateAfterBindContractTests.cpp
        tests/vulkan/commands/VulkanBarrierContractTests.cpp
        tests/vulkan/commands/VulkanCommandPoolContractTests.cpp
        tests/vulkan/commands/VulkanQueueContractTests.cpp
        tests/vulkan/compute/VulkanComputeContractTests.cpp
        tests/vulkan/compute/VulkanIndirectDispatchBatchContractTests.cpp
        tests/vulkan/device/VulkanBackendDispatchContractTests.cpp
        tests/vulkan/device/VulkanDeviceLossContractTests.cpp
        tests/vulkan/device/VulkanHandleProvenanceContractTests.cpp
        tests/vulkan/device/VulkanHeadlessDeviceContractTests.cpp
        tests/vulkan/device/VulkanLinuxProfileContractTests.cpp
        tests/vulkan/device/VulkanMacosProfileContractTests.cpp
        tests/vulkan/device/VulkanWindowsProfileContractTests.cpp
        tests/vulkan/diagnostics/VulkanDiagnosticsContractTests.cpp
        tests/vulkan/internals/VulkanCachingAndRetentionContractTests.cpp
        tests/vulkan/internals/VulkanDispatchTableContractTests.cpp
        tests/vulkan/internals/VulkanOptionalFeatureContractTests.cpp
        tests/vulkan/internals/VulkanValidUsageContractTests.cpp
        tests/vulkan/interop/VulkanSharedInteropContractTests.cpp
        tests/vulkan/memory/VulkanPlacedHeapContractTests.cpp
        tests/vulkan/memory/VulkanResidencyContractTests.cpp
        tests/vulkan/pipelines/VulkanPipelineCacheContractTests.cpp
        tests/vulkan/pipelines/VulkanPipelineContractTests.cpp
        tests/vulkan/pipelines/VulkanStageLinkageContractTests.cpp
        tests/vulkan/presentation/VulkanAcquireTeardownContractTests.cpp
        tests/vulkan/presentation/VulkanLinuxSurfaceContractTests.cpp
        tests/vulkan/presentation/VulkanPresentSyncContractTests.cpp
        tests/vulkan/presentation/VulkanResizeLifecycleContractTests.cpp
        tests/vulkan/presentation/VulkanSurfaceColorContractTests.cpp
        tests/vulkan/presentation/VulkanSurfaceContractTests.cpp
        tests/vulkan/queries/VulkanConditionalExecutionContractTests.cpp
        tests/vulkan/queries/VulkanQueryContractTests.cpp
        tests/vulkan/raytracing/VulkanRayQueryContractTests.cpp
        tests/vulkan/raytracing/VulkanRayTracingPipelineContractTests.cpp
        tests/vulkan/raytracing/VulkanTopLevelInstanceBufferContractTests.cpp
        tests/vulkan/rendering/VulkanGraphicsContractTests.cpp
        tests/vulkan/rendering/VulkanDrawIndexContractTests.cpp
        tests/vulkan/rendering/VulkanHeadlessParityContractTests.cpp
        tests/vulkan/rendering/VulkanIndirectContractTests.cpp
        tests/vulkan/rendering/VulkanMeshShaderContractTests.cpp
        tests/vulkan/rendering/VulkanRasterContractTests.cpp
        tests/vulkan/rendering/VulkanRasterizerOrderedStorageContractTests.cpp
        tests/vulkan/rendering/VulkanVariableRateShadingContractTests.cpp
        tests/vulkan/resources/VulkanBufferContractTests.cpp
        tests/vulkan/resources/VulkanCopiesContractTests.cpp
        tests/vulkan/resources/VulkanFillAndClearContractTests.cpp
        tests/vulkan/resources/VulkanImageUtilityContractTests.cpp
        tests/vulkan/resources/VulkanMipmapSrgbContractTests.cpp
        tests/vulkan/resources/VulkanTextureContractTests.cpp
        tests/vulkan/shaders/VulkanShaderArithmeticContractTests.cpp
        tests/vulkan/shaders/VulkanShaderModuleContractTests.cpp
    )
    if(NOT RACHIS_ENABLE_VULKAN)
        get_target_property(rachis_d3d12_test_sources RachisRHITests SOURCES)
        set(rachis_vulkan_internal_tests ${rachis_d3d12_test_sources})
        list(FILTER rachis_vulkan_internal_tests INCLUDE REGEX "(^|/)tests/vulkan/internals/")
        list(FILTER rachis_d3d12_test_sources EXCLUDE REGEX "(^|/)tests/vulkan/")
        set_property(TARGET RachisRHITests PROPERTY SOURCES
            ${rachis_d3d12_test_sources} ${rachis_vulkan_internal_tests})
    endif()
    target_include_directories(RachisRHITests PRIVATE "${rachis_rhi_source_dir}")
    target_compile_definitions(RachisRHITests
        PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX
            NOGDI
            RACHIS_TEST_COMPUTE_SHADER_PATH="${rachis_rhi_dxil_dir}/ComputeWrite.dxil"
            RACHIS_TEST_INDIRECT_DISPATCH_BATCH_SHADER_PATH="${rachis_rhi_dxil_dir}/IndirectDispatchBatch.dxil"
            RACHIS_TEST_RAY_QUERY_SHADER_PATH="${rachis_rhi_dxil_dir}/RayQuery.dxil"
            RACHIS_TEST_INSTANCE_RECORD_WRITER_SHADER_PATH="${rachis_rhi_dxil_dir}/InstanceRecordWriter.dxil"
            RACHIS_TEST_INSTANCE_BUFFER_RAY_QUERY_SHADER_PATH="${rachis_rhi_dxil_dir}/InstanceBufferRayQuery.dxil"
            RACHIS_TEST_RAY_TRACING_PIPELINE_SHADER_PATH="${rachis_rhi_dxil_dir}/RayTracingPipeline.dxil"
            RACHIS_TEST_CONDITIONAL_EXECUTION_SHADER_PATH="${rachis_rhi_dxil_dir}/ConditionalExecution.dxil"
            RACHIS_TEST_SPECIALIZATION_VARIANT_7_PATH="${rachis_rhi_dxil_dir}/SpecializationVariant7.dxil"
            RACHIS_TEST_SPECIALIZATION_VARIANT_13_PATH="${rachis_rhi_dxil_dir}/SpecializationVariant13.dxil"
            RACHIS_TEST_SHADER_16_BIT_PATH="${rachis_rhi_dxil_dir}/Shader16Bit.dxil"
            RACHIS_TEST_SHADER_INT64_PATH="${rachis_rhi_dxil_dir}/ShaderInt64.dxil"
            RACHIS_TEST_SHADER_INT64_ATOMIC_GROUP_SHARED_PATH="${rachis_rhi_dxil_dir}/ShaderInt64AtomicGroupShared.dxil"
            RACHIS_TEST_SHADER_INT64_ATOMIC_TYPED_RESOURCE_PATH="${rachis_rhi_dxil_dir}/ShaderInt64AtomicTypedResource.dxil"
            RACHIS_TEST_SUBGROUP_OPERATIONS_SHADER_PATH="${rachis_rhi_dxil_dir}/SubgroupOperations.dxil"
            RACHIS_TEST_BINDING_BATCH_SHADER_PATH="${rachis_rhi_dxil_dir}/BindingBatch.dxil"
            RACHIS_TEST_DYNAMIC_BINDING_SHADER_PATH="${rachis_rhi_dxil_dir}/DynamicBinding.dxil"
            RACHIS_TEST_DYNAMIC_BINDING_COUNTER_SHADER_PATH="${rachis_rhi_dxil_dir}/DynamicBindingCounter.dxil"
            RACHIS_TEST_UNIFORM_BUFFER_RANGE_SHADER_PATH="${rachis_rhi_dxil_dir}/UniformBufferRange.dxil"
            RACHIS_TEST_DYNAMIC_UNIFORM_BUFFER_ARRAY_SHADER_PATH="${rachis_rhi_dxil_dir}/DynamicUniformBufferArray.dxil"
            RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/GraphicsDirect.vs.dxil"
            RACHIS_TEST_GRAPHICS_SHADING_RATE_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/GraphicsShadingRate.vs.dxil"
            RACHIS_TEST_GRAPHICS_INPUT_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/GraphicsInput.vs.dxil"
            RACHIS_TEST_GRAPHICS_UNCLIPPED_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/GraphicsUnclipped.vs.dxil"
            RACHIS_TEST_GRAPHICS_DEPTH_BIAS_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/GraphicsDepthBias.vs.dxil"
            RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/Graphics.ps.dxil"
            RACHIS_TEST_TRIANGLE_FAN_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/TriangleFan.vs.dxil"
            RACHIS_TEST_TRIANGLE_FAN_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/TriangleFan.ps.dxil"
            RACHIS_TEST_SAMPLER_FILTER_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/SamplerFilter.vs.dxil"
            RACHIS_TEST_SAMPLER_FILTER_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/SamplerFilter.ps.dxil"
            RACHIS_TEST_LOGIC_OPERATION_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/LogicOperation.ps.dxil"
            RACHIS_TEST_GEOMETRY_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/GeometryShader.vs.dxil"
            RACHIS_TEST_GEOMETRY_SHADER_PATH="${rachis_rhi_dxil_dir}/GeometryShader.gs.dxil"
            RACHIS_TEST_MULTI_VIEWPORT_GEOMETRY_SHADER_PATH="${rachis_rhi_dxil_dir}/MultiViewport.gs.dxil"
            RACHIS_TEST_GEOMETRY_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/GeometryShader.ps.dxil"
            RACHIS_TEST_TESSELLATION_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/TessellationShader.vs.dxil"
            RACHIS_TEST_TESSELLATION_HULL_SHADER_PATH="${rachis_rhi_dxil_dir}/TessellationShader.hs.dxil"
            RACHIS_TEST_TESSELLATION_DOMAIN_SHADER_PATH="${rachis_rhi_dxil_dir}/TessellationShader.ds.dxil"
            RACHIS_TEST_TESSELLATION_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/TessellationShader.ps.dxil"
            RACHIS_TEST_TASK_SHADER_PATH="${rachis_rhi_dxil_dir}/MeshShader.as.dxil"
            RACHIS_TEST_MESH_SHADER_PATH="${rachis_rhi_dxil_dir}/MeshShader.ms.dxil"
            RACHIS_TEST_MESH_NO_TASK_SHADER_PATH="${rachis_rhi_dxil_dir}/MeshShader.no-task.ms.dxil"
            RACHIS_TEST_MESH_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/MeshShader.ps.dxil"
            RACHIS_TEST_DRAW_INDEX_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/DrawIndex.vs.dxil"
            RACHIS_TEST_DRAW_INDEX_PLAIN_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/DrawIndex.plain.vs.dxil"
            RACHIS_TEST_DRAW_INDEX_MESH_SHADER_PATH="${rachis_rhi_dxil_dir}/DrawIndex.ms.dxil"
            RACHIS_TEST_DRAW_INDEX_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/DrawIndex.ps.dxil"
            RACHIS_TEST_DUAL_SOURCE_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/DualSourceBlend.ps.dxil"
            RACHIS_TEST_RASTERIZER_ORDERED_STORAGE_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/RasterizerOrderedStorage.ps.dxil"
            RACHIS_TEST_FRAGMENT_PRIMITIVE_ID_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/FragmentPrimitiveId.vs.dxil"
            RACHIS_TEST_FRAGMENT_PRIMITIVE_ID_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/FragmentPrimitiveId.ps.dxil"
            RACHIS_TEST_FRAGMENT_BARYCENTRIC_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/FragmentBarycentric.vs.dxil"
            RACHIS_TEST_FRAGMENT_BARYCENTRIC_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/FragmentBarycentric.ps.dxil"
            RACHIS_TEST_LAYERED_RENDERING_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/LayeredRendering.vs.dxil"
            RACHIS_TEST_LAYERED_RENDERING_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/LayeredRendering.ps.dxil"
            RACHIS_TEST_MULTIVIEW_RENDERING_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/MultiviewRendering.vs.dxil"
            RACHIS_TEST_MULTIVIEW_RENDERING_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/MultiviewRendering.ps.dxil"
            RACHIS_TEST_VERTEX_INPUT_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/VertexInput.vs.dxil"
            RACHIS_TEST_VERTEX_INPUT_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/VertexInput.ps.dxil"
            RACHIS_TEST_VERTEX_INPUT_MATRIX_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/VertexInput.matrix.vs.dxil"
            RACHIS_TEST_VERTEX_INPUT_ARRAY_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/VertexInput.array.vs.dxil"
            RACHIS_TEST_PRIMITIVE_RESTART_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/PrimitiveRestart.vs.dxil"
            RACHIS_TEST_PRIMITIVE_RESTART_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/PrimitiveRestart.ps.dxil"
            RACHIS_TEST_POINT_LIST_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/PointList.vs.dxil"
            RACHIS_TEST_POINT_LIST_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/PointList.ps.dxil"
            RACHIS_TEST_SHADER_DISTANCE_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/ShaderDistance.vs.dxil"
            RACHIS_TEST_SHADER_DISTANCE_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/ShaderDistance.ps.dxil"
            RACHIS_TEST_MIPMAP_AFTER_RENDERING_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/MipmapAfterRendering.vs.dxil"
            RACHIS_TEST_MIPMAP_AFTER_RENDERING_SLOW_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/MipmapAfterRendering.slow.ps.dxil"
            RACHIS_TEST_MIPMAP_AFTER_RENDERING_BLEND_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/MipmapAfterRendering.blend.ps.dxil"
            RACHIS_TEST_INDIRECT_COMPUTE_SHADER_PATH="${rachis_rhi_dxil_dir}/IndirectArguments.dxil"
            RACHIS_TEST_PARTIALLY_BOUND_BINDING_SHADER_PATH="${rachis_rhi_dxil_dir}/PartiallyBoundBinding.dxil"
            RACHIS_TEST_VARIABLE_BINDING_SHADER_PATH="${rachis_rhi_dxil_dir}/VariableBinding.dxil"
            RACHIS_TEST_UPDATE_AFTER_BIND_BINDING_SHADER_PATH="${rachis_rhi_dxil_dir}/UpdateAfterBindBinding.dxil"
            RACHIS_TEST_TEXTURE_VIEW_COMPONENT_SWIZZLE_SHADER_PATH="${rachis_rhi_dxil_dir}/TextureViewComponentSwizzle.dxil"
            RACHIS_TEST_MIPMAP_SHADER_READ_COMPUTE_SHADER_PATH="${rachis_rhi_dxil_dir}/MipmapShaderRead.dxil"
            RACHIS_TEST_MIPMAP_SHADER_READ_VERTEX_SHADER_PATH="${rachis_rhi_dxil_dir}/MipmapShaderRead.vs.dxil"
            RACHIS_TEST_MIPMAP_SHADER_READ_PIXEL_SHADER_PATH="${rachis_rhi_dxil_dir}/MipmapShaderRead.ps.dxil"
            RACHIS_TEST_TYPED_INTEGER_TEXTURE_SHADER_PATH="${rachis_rhi_dxil_dir}/TypedIntegerTexture.dxil"
            RACHIS_TEST_TYPED_INTEGER_SAMPLED_LOAD_SHADER_PATH="${rachis_rhi_dxil_dir}/TypedIntegerSampledLoad.dxil"
            RACHIS_TEST_BLOCK_COMPRESSED_TEXTURE_SHADER_PATH="${rachis_rhi_dxil_dir}/BlockCompressedTexture.dxil"
            RACHIS_TEST_SAMPLER_REDUCTION_SHADER_PATH="${rachis_rhi_dxil_dir}/SamplerReduction.dxil"
            RACHIS_TEST_SPARSE_BUFFER_SHADER_PATH="${rachis_rhi_dxil_dir}/SparseBuffer.dxil"
            RACHIS_TEST_SPARSE_TEXTURE_SHADER_PATH="${rachis_rhi_dxil_dir}/SparseTexture.dxil"
            RACHIS_TEST_MIP_REGION_USED_SAMPLER_FEEDBACK_SHADER_PATH="${rachis_rhi_dxil_dir}/MipRegionUsedSamplerFeedback.dxil"
            RACHIS_TEST_MIN_MIP_SAMPLER_FEEDBACK_SHADER_PATH="${rachis_rhi_dxil_dir}/MinMipSamplerFeedback.dxil"
    )
    target_link_libraries(RachisRHITests PRIVATE Rachis::RHI RachisRHITestSupport RachisRHITestHooks RachisRHITestMain Catch2::Catch2)
    target_compile_definitions(RachisRHITests
        PRIVATE
            RACHIS_ENABLE_D3D12=1
            RACHIS_ENABLE_VULKAN=$<BOOL:${RACHIS_ENABLE_VULKAN}>
            RACHIS_TESTING
    )
    if(RACHIS_ENABLE_VULKAN)
        target_link_libraries(RachisRHITests PRIVATE RachisRHIVulkan volk Vulkan-Headers VulkanMemoryAllocator)
        target_compile_definitions(RachisRHITests PRIVATE VK_NO_PROTOTYPES)
        if(TARGET RachisRHITestSpirvShaders)
            add_dependencies(RachisRHITests RachisRHITestSpirvShaders)
            target_compile_definitions(RachisRHITests PRIVATE
                RACHIS_SPIRV_DIR="${rachis_rhi_spirv_dir}"
                RACHIS_RENDERER_SPIRV_DIR="${rachis_rhi_spirv_dir}"
                RACHIS_PARITY_EVIDENCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/parity-evidence"
                RACHIS_SLANGC_EXECUTABLE="${RACHIS_SLANGC_EXECUTABLE}"
            )
        endif()
    endif()
    rachis_configure_rhi_target(RachisRHITests)
    if(TARGET RachisWinPixEventRuntimeCopy)
        add_dependencies(RachisRHITests RachisWinPixEventRuntimeCopy)
    endif()
    rachis_rhi_test_dxil(SubgroupOperations.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SubgroupOperations.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(ComputeWrite.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ComputeWrite.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(SparseBuffer.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SparseBuffer.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(SparseTexture.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SparseTexture.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(MipRegionUsedSamplerFeedback.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipRegionUsedSamplerFeedback.hlsl" cs_6_5 ENTRY main)
    rachis_rhi_test_dxil(MinMipSamplerFeedback.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MinMipSamplerFeedback.hlsl" cs_6_5 ENTRY main)
    rachis_rhi_test_dxil(IndirectDispatchBatch.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ComputeWrite.hlsl" cs_6_0 ENTRY countDispatch)
    rachis_rhi_test_dxil(TypedIntegerTexture.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TypedIntegerTexture.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(TypedIntegerSampledLoad.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TypedIntegerSampledLoad.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(BlockCompressedTexture.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BlockCompressedTexture.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(SamplerReduction.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SamplerReduction.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(RayQuery.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RayQuery.hlsl" cs_6_5 ENTRY main)
    rachis_rhi_test_dxil(InstanceRecordWriter.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/InstanceRecordWriter.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(InstanceBufferRayQuery.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/InstanceBufferRayQuery.hlsl" cs_6_5 ENTRY main)
    rachis_rhi_test_dxil(RayTracingPipeline.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RayTracingPipeline.hlsl" lib_6_3)
    rachis_rhi_test_dxil(ConditionalExecution.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ConditionalExecution.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(SpecializationVariant7.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SpecializationVariant.hlsl" cs_6_0 ENTRY main OPTIONS -D RACHIS_SPECIALIZED_VALUE=7)
    rachis_rhi_test_dxil(SpecializationVariant13.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SpecializationVariant.hlsl" cs_6_0 ENTRY main OPTIONS -D RACHIS_SPECIALIZED_VALUE=13)
    rachis_rhi_test_dxil(Shader16Bit.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/Shader16Bit.hlsl" cs_6_2 ENTRY main OPTIONS -Od -enable-16bit-types)
    rachis_rhi_test_dxil(ShaderInt64.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64.hlsl" cs_6_0 ENTRY main OPTIONS -Od)
    rachis_rhi_test_dxil(ShaderInt64AtomicGroupShared.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64AtomicGroupShared.hlsl" cs_6_6 ENTRY main OPTIONS -Od)
    rachis_rhi_test_dxil(ShaderInt64AtomicTypedResource.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderInt64AtomicTypedResource.hlsl" cs_6_6 ENTRY main OPTIONS -Od)
    rachis_rhi_test_dxil(BindingBatch.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/BindingBatch.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(IndirectArguments.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/IndirectArguments.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(DynamicBinding.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DynamicBinding.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(DynamicBindingCounter.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DynamicBindingCounter.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(UniformBufferRange.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/UniformBufferRange.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(DynamicUniformBufferArray.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DynamicUniformBufferArray.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(PartiallyBoundBinding.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PartiallyBoundBinding.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(VariableBinding.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VariableBinding.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(UpdateAfterBindBinding.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/UpdateAfterBindBinding.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(TextureViewComponentSwizzle.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TextureViewComponentSwizzle.hlsl" cs_6_0 ENTRY main)
    rachis_rhi_test_dxil(MipmapShaderRead.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapShaderRead.hlsl" cs_6_0 ENTRY computeMain)
    rachis_rhi_test_dxil(GraphicsDirect.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl" vs_6_0 ENTRY directVertex)
    rachis_rhi_test_dxil(GraphicsShadingRate.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl" vs_6_4 ENTRY shadingRateVertex)
    rachis_rhi_test_dxil(GraphicsInput.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl" vs_6_0 ENTRY inputVertex)
    rachis_rhi_test_dxil(GraphicsUnclipped.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl" vs_6_0 ENTRY unclippedVertex)
    rachis_rhi_test_dxil(GraphicsDepthBias.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl" vs_6_0 ENTRY depthBiasVertex)
    rachis_rhi_test_dxil(Graphics.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GraphicsTriangle.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(MipmapShaderRead.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapShaderRead.hlsl" vs_6_0 ENTRY fullscreenVertex)
    rachis_rhi_test_dxil(MipmapShaderRead.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapShaderRead.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(TriangleFan.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TriangleFan.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(TriangleFan.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TriangleFan.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(SamplerFilter.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SamplerFilter.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(SamplerFilter.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/SamplerFilter.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(LogicOperation.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LogicOperation.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(GeometryShader.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl" vs_6_0 ENTRY pointVertex)
    rachis_rhi_test_dxil(GeometryShader.gs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl" gs_6_0 ENTRY geometryResourceMain)
    rachis_rhi_test_dxil(MultiViewport.gs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl" gs_6_0 ENTRY viewportArrayMain)
    rachis_rhi_test_dxil(GeometryShader.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/GeometryShader.hlsl" ps_6_0 ENTRY geometryPixelMain)
    rachis_rhi_test_dxil(TessellationShader.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl" vs_6_0 ENTRY tessellationVertex)
    rachis_rhi_test_dxil(TessellationShader.hs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl" hs_6_0 ENTRY tessellationHull)
    rachis_rhi_test_dxil(TessellationShader.ds.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl" ds_6_0 ENTRY tessellationDomain)
    rachis_rhi_test_dxil(TessellationShader.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/TessellationShader.hlsl" ps_6_0 ENTRY tessellationPixel)
    rachis_rhi_test_dxil(MeshShader.as.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MeshShader.hlsl" as_6_5 ENTRY taskMain)
    rachis_rhi_test_dxil(MeshShader.ms.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MeshShader.hlsl" ms_6_5 ENTRY meshMain)
    rachis_rhi_test_dxil(MeshShader.no-task.ms.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MeshShader.hlsl" ms_6_5 ENTRY meshNoTask)
    rachis_rhi_test_dxil(MeshShader.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MeshShader.hlsl" ps_6_5 ENTRY meshPixel)
    foreach(rachis_draw_index_entry IN ITEMS
            drawIndexVertex:vs_6_0:DrawIndex.vs plainVertex:vs_6_0:DrawIndex.plain.vs
            drawIndexMesh:ms_6_5:DrawIndex.ms recordPixel:ps_6_5:DrawIndex.ps)
        string(REPLACE ":" ";" rachis_draw_index_parts "${rachis_draw_index_entry}")
        list(GET rachis_draw_index_parts 0 rachis_draw_index_name)
        list(GET rachis_draw_index_parts 1 rachis_draw_index_profile)
        list(GET rachis_draw_index_parts 2 rachis_draw_index_output)
        rachis_rhi_test_dxil(${rachis_draw_index_output}.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DrawIndex.hlsl"
            ${rachis_draw_index_profile} ENTRY ${rachis_draw_index_name}
            INCLUDE_DIRECTORIES "${rachis_rhi_public_include_dir}")
    endforeach()
    rachis_rhi_test_dxil(DualSourceBlend.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/DualSourceBlend.hlsl" ps_6_0 ENTRY main)
    rachis_rhi_test_dxil(RasterizerOrderedStorage.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/RasterizerOrderedStorage.hlsl" ps_6_0 ENTRY main)
    rachis_rhi_test_dxil(FragmentPrimitiveId.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl" vs_6_0 ENTRY primitiveIdVertex)
    rachis_rhi_test_dxil(FragmentPrimitiveId.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl" ps_6_0 ENTRY primitiveIdPixel)
    rachis_rhi_test_dxil(FragmentBarycentric.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl" vs_6_0 ENTRY barycentricVertex)
    rachis_rhi_test_dxil(FragmentBarycentric.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/FragmentShaderBuiltins.hlsl" ps_6_1 ENTRY barycentricPixel)
    rachis_rhi_test_dxil(LayeredRendering.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LayeredRendering.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(LayeredRendering.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/LayeredRendering.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(MultiviewRendering.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MultiviewRendering.hlsl" vs_6_1 ENTRY vsMain)
    rachis_rhi_test_dxil(MultiviewRendering.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MultiviewRendering.hlsl" ps_6_0 ENTRY psMain)
    rachis_rhi_test_dxil(VertexInput.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(VertexInput.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(VertexInput.matrix.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl" vs_6_0 ENTRY matrixVertexMain)
    rachis_rhi_test_dxil(VertexInput.array.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/VertexInput.hlsl" vs_6_0 ENTRY arrayVertexMain)
    rachis_rhi_test_dxil(PrimitiveRestart.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PrimitiveRestart.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(PrimitiveRestart.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PrimitiveRestart.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(PointList.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PointList.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(PointList.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PointList.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(ShaderDistance.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderDistance.hlsl" vs_6_0 ENTRY vertexMain)
    rachis_rhi_test_dxil(ShaderDistance.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/ShaderDistance.hlsl" ps_6_0 ENTRY pixelMain)
    rachis_rhi_test_dxil(MipmapAfterRendering.vs.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl" vs_6_0 ENTRY fullscreenVertex)
    rachis_rhi_test_dxil(MipmapAfterRendering.slow.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl" ps_6_0 ENTRY slowPixel)
    rachis_rhi_test_dxil(MipmapAfterRendering.blend.ps.dxil "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/MipmapAfterRendering.hlsl" ps_6_0 ENTRY blendPixel)
    add_custom_target(RachisRHITestDxilShaders ALL DEPENDS ${rachis_rhi_test_dxil_outputs})
    add_dependencies(RachisRHITests RachisRHITestDxilShaders)
    catch_discover_tests(
        RachisRHITests
        TEST_PREFIX "RachisRHI."
        DISCOVERY_MODE PRE_TEST
        ADD_TAGS_AS_LABELS
        PROPERTIES TIMEOUT 30
    )
    elseif(RACHIS_ENABLE_VULKAN)
        set(rachis_rhi_vulkan_tests
            tests/core/FrontendDefaultsAndInternalsContractTests.cpp
            tests/core/FrontendValidationContractTests.cpp
            tests/core/HandleTableTests.cpp
            tests/core/PipelineLayoutDerivationTests.cpp
            tests/core/RayTracingShaderTableLayoutTests.cpp
            tests/core/ShaderBinaryToolContractTests.cpp
            tests/core/ShaderInterfaceReflectionTests.cpp
            tests/core/StageLinkageTests.cpp
            tests/d3d12/internals/D3D12AccelerationStructureLeaseTests.cpp
            tests/d3d12/internals/D3D12BarrierTranslationTests.cpp
            tests/d3d12/internals/D3D12DxilSignatureTests.cpp
            tests/d3d12/internals/D3D12RuntimePoliciesTests.cpp
            tests/d3d12/internals/D3D12SwapchainLeaseTests.cpp
            tests/metal/commands/MetalBarrierContractTests.cpp
            tests/metal/commands/MetalCommandPoolContractTests.cpp
            tests/metal/commands/MetalQueueContractTests.cpp
            tests/metal/device/MetalBackendDispatchContractTests.cpp
            tests/metal/device/MetalDeviceLossContractTests.cpp
            tests/metal/device/MetalHandleProvenanceContractTests.cpp
            tests/metal/device/MetalHeadlessDeviceContractTests.cpp
            tests/metal/device/MetalLifecycleContractTests.cpp
            tests/metal/device/MetalMacosProfileContractTests.cpp
            tests/metal/diagnostics/MetalDiagnosticsContractTests.cpp
            tests/metal/memory/MetalBudgetContractTests.cpp
            tests/metal/resources/MetalBufferContractTests.cpp
            tests/metal/resources/MetalCopiesContractTests.cpp
            tests/metal/resources/MetalTextureContractTests.cpp
            tests/portable/PortableBackendContractTests.cpp
            tests/portable/GeneratedShaderInterfaceTests.cpp
            tests/vulkan/binding/VulkanBindingContractTests.cpp
            tests/vulkan/binding/VulkanDescriptorIndexingContractTests.cpp
            tests/vulkan/binding/VulkanUpdateAfterBindContractTests.cpp
            tests/vulkan/commands/VulkanBarrierContractTests.cpp
            tests/vulkan/commands/VulkanCommandPoolContractTests.cpp
            tests/vulkan/commands/VulkanQueueContractTests.cpp
            tests/vulkan/compute/VulkanComputeContractTests.cpp
            tests/vulkan/compute/VulkanIndirectDispatchBatchContractTests.cpp
            tests/vulkan/device/VulkanBackendDispatchContractTests.cpp
            tests/vulkan/device/VulkanDeviceLossContractTests.cpp
            tests/vulkan/device/VulkanHandleProvenanceContractTests.cpp
            tests/vulkan/device/VulkanHeadlessDeviceContractTests.cpp
            tests/vulkan/device/VulkanLinuxProfileContractTests.cpp
            tests/vulkan/device/VulkanMacosProfileContractTests.cpp
            tests/vulkan/device/VulkanWindowsProfileContractTests.cpp
            tests/vulkan/diagnostics/VulkanDiagnosticsContractTests.cpp
            tests/vulkan/internals/VulkanCachingAndRetentionContractTests.cpp
            tests/vulkan/internals/VulkanDispatchTableContractTests.cpp
            tests/vulkan/internals/VulkanOptionalFeatureContractTests.cpp
            tests/vulkan/internals/VulkanValidUsageContractTests.cpp
            tests/vulkan/interop/VulkanSharedInteropContractTests.cpp
            tests/vulkan/memory/VulkanPlacedHeapContractTests.cpp
            tests/vulkan/memory/VulkanResidencyContractTests.cpp
            tests/vulkan/pipelines/VulkanPipelineCacheContractTests.cpp
            tests/vulkan/pipelines/VulkanPipelineContractTests.cpp
            tests/vulkan/pipelines/VulkanStageLinkageContractTests.cpp
            tests/vulkan/presentation/VulkanAcquireTeardownContractTests.cpp
            tests/vulkan/presentation/VulkanLinuxSurfaceContractTests.cpp
            tests/vulkan/queries/VulkanConditionalExecutionContractTests.cpp
            tests/vulkan/queries/VulkanQueryContractTests.cpp
            tests/vulkan/raytracing/VulkanRayQueryContractTests.cpp
            tests/vulkan/raytracing/VulkanRayTracingPipelineContractTests.cpp
            tests/vulkan/raytracing/VulkanTopLevelInstanceBufferContractTests.cpp
            tests/vulkan/rendering/VulkanGraphicsContractTests.cpp
            tests/vulkan/rendering/VulkanDrawIndexContractTests.cpp
            tests/vulkan/rendering/VulkanHeadlessParityContractTests.cpp
            tests/vulkan/rendering/VulkanIndirectContractTests.cpp
            tests/vulkan/rendering/VulkanMeshShaderContractTests.cpp
            tests/vulkan/rendering/VulkanRasterContractTests.cpp
            tests/vulkan/rendering/VulkanRasterizerOrderedStorageContractTests.cpp
            tests/vulkan/rendering/VulkanVariableRateShadingContractTests.cpp
            tests/vulkan/resources/VulkanBufferContractTests.cpp
            tests/vulkan/resources/VulkanCopiesContractTests.cpp
            tests/vulkan/resources/VulkanFillAndClearContractTests.cpp
            tests/vulkan/resources/VulkanImageUtilityContractTests.cpp
            tests/vulkan/resources/VulkanMipmapSrgbContractTests.cpp
            tests/vulkan/resources/VulkanTextureContractTests.cpp
            tests/vulkan/shaders/VulkanShaderArithmeticContractTests.cpp
            tests/vulkan/shaders/VulkanShaderModuleContractTests.cpp
        )
        if(WIN32)
            list(APPEND rachis_rhi_vulkan_tests
                tests/vulkan/presentation/VulkanPresentSyncContractTests.cpp
                tests/vulkan/presentation/VulkanResizeLifecycleContractTests.cpp
                tests/vulkan/presentation/VulkanSurfaceColorContractTests.cpp
                tests/vulkan/presentation/VulkanSurfaceContractTests.cpp
            )
        elseif(UNIX AND NOT APPLE)
            list(APPEND rachis_rhi_vulkan_tests tests/vulkan/presentation/VulkanLinuxPresentContractTests.cpp)
        endif()
        add_executable(RachisRHITests ${rachis_rhi_vulkan_tests})
        target_include_directories(RachisRHITests PRIVATE "${rachis_rhi_source_dir}")
        target_compile_definitions(RachisRHITests
            PRIVATE
                RACHIS_ENABLE_D3D12=0
                RACHIS_ENABLE_VULKAN=1
                RACHIS_TESTING
                VK_NO_PROTOTYPES
                VK_ENABLE_BETA_EXTENSIONS
        )
        if(WIN32)
            target_compile_definitions(RachisRHITests PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX NOGDI)
        endif()
        target_link_libraries(RachisRHITests PRIVATE Rachis::RHI RachisRHITestSupport RachisRHITestHooks RachisRHITestMain Catch2::Catch2 volk Vulkan-Headers VulkanMemoryAllocator)
        if(TARGET RachisRHILinuxSurfaceTestHost)
            target_link_libraries(RachisRHITests PRIVATE RachisRHILinuxSurfaceTestHost)
        endif()
        if(TARGET RachisRHITestSpirvShaders)
            add_dependencies(RachisRHITests RachisRHITestSpirvShaders)
            target_compile_definitions(RachisRHITests PRIVATE
                RACHIS_SPIRV_DIR="${rachis_rhi_spirv_dir}"
                RACHIS_RENDERER_SPIRV_DIR="${rachis_rhi_spirv_dir}"
                RACHIS_PARITY_EVIDENCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/parity-evidence"
                RACHIS_SLANGC_EXECUTABLE="${RACHIS_SLANGC_EXECUTABLE}"
            )
        endif()
        rachis_configure_rhi_target(RachisRHITests)
        if(APPLE)
            set(rachis_rhi_tests_rpath "@loader_path")
            if(RACHIS_ENABLE_VULKAN)
                rachis_prepend_vulkan_sdk_rpath(rachis_rhi_tests_rpath)
            endif()
            set_target_properties(RachisRHITests PROPERTIES
                MACOSX_RPATH ON
                BUILD_RPATH "${rachis_rhi_tests_rpath}"
            )
        endif()
        catch_discover_tests(
            RachisRHITests
            TEST_PREFIX "RachisRHI."
            DISCOVERY_MODE PRE_TEST
            ADD_TAGS_AS_LABELS
            PROPERTIES TIMEOUT 30
        )
    elseif(RACHIS_ENABLE_METAL)
        add_executable(RachisRHITests
            tests/core/FrontendDefaultsAndInternalsContractTests.cpp
            tests/core/FrontendValidationContractTests.cpp
            tests/core/HandleTableTests.cpp
            tests/core/PipelineLayoutDerivationTests.cpp
            tests/core/RayTracingShaderTableLayoutTests.cpp
            tests/core/ShaderBinaryToolContractTests.cpp
            tests/core/ShaderInterfaceReflectionTests.cpp
            tests/core/StageLinkageTests.cpp
            tests/d3d12/internals/D3D12AccelerationStructureLeaseTests.cpp
            tests/d3d12/internals/D3D12BarrierTranslationTests.cpp
            tests/d3d12/internals/D3D12DxilSignatureTests.cpp
            tests/d3d12/internals/D3D12RuntimePoliciesTests.cpp
            tests/d3d12/internals/D3D12SwapchainLeaseTests.cpp
            tests/metal/binding/MetalBindingArrayContractTests.cpp
            tests/metal/binding/MetalBindingContractTests.cpp
            tests/metal/binding/MetalUpdateAfterBindContractTests.cpp
            tests/metal/commands/MetalBarrierContractTests.cpp
            tests/metal/commands/MetalCommandPoolContractTests.cpp
            tests/metal/commands/MetalQueueContractTests.cpp
            tests/metal/commands/MetalTextureLayoutContractTests.cpp
            tests/metal/compute/MetalComputeContractTests.cpp
            tests/metal/device/MetalBackendDispatchContractTests.cpp
            tests/metal/device/MetalDeviceLossContractTests.cpp
            tests/metal/device/MetalHandleProvenanceContractTests.cpp
            tests/metal/device/MetalHeadlessDeviceContractTests.cpp
            tests/metal/device/MetalLifecycleContractTests.cpp
            tests/metal/device/MetalMacosProfileContractTests.cpp
            tests/metal/diagnostics/MetalDiagnosticsContractTests.cpp
            tests/metal/memory/MetalBudgetContractTests.cpp
            tests/metal/memory/MetalPlacedHeapContractTests.cpp
            tests/metal/performance/MetalResidencyAndPoolingContractTests.cpp
            tests/metal/pipelines/MetalPipelineCacheContractTests.cpp
            tests/metal/pipelines/MetalPipelineContractTests.cpp
            tests/metal/pipelines/MetalStageLinkageContractTests.cpp
            tests/metal/presentation/MetalPresentContractTests.cpp
            tests/metal/presentation/MetalPresentationLifecycleContractTests.cpp
            tests/metal/presentation/MetalResizeLifecycleContractTests.cpp
            tests/metal/presentation/MetalSurfaceColorContractTests.cpp
            tests/metal/presentation/MetalSurfaceContractTests.cpp
            tests/metal/queries/MetalQueryContractTests.cpp
            tests/metal/raytracing/MetalRayTracingContractTests.cpp
            tests/metal/rendering/MetalHeadlessParityContractTests.cpp
            tests/metal/rendering/MetalDrawIndexContractTests.cpp
            tests/metal/rendering/MetalIndirectContractTests.cpp
            tests/metal/rendering/MetalLayeredRenderingContractTests.cpp
            tests/metal/rendering/MetalMeshShaderContractTests.cpp
            tests/metal/rendering/MetalRasterContractTests.cpp
            tests/metal/rendering/MetalUnsupportedRasterFeatureContractTests.cpp
            tests/metal/resources/MetalBufferContractTests.cpp
            tests/metal/resources/MetalCopiesContractTests.cpp
            tests/metal/resources/MetalFillAndClearContractTests.cpp
            tests/metal/resources/MetalImageUtilityContractTests.cpp
            tests/metal/resources/MetalTextureContractTests.cpp
            tests/metal/shaders/MetalShaderArithmeticContractTests.cpp
            tests/metal/shaders/MetalShaderCookingContractTests.cpp
            tests/metal/shaders/MetalShaderModuleContractTests.cpp
            tests/portable/PortableBackendContractTests.cpp
            tests/portable/GeneratedShaderInterfaceTests.cpp
            tests/vulkan/internals/VulkanCachingAndRetentionContractTests.cpp
            tests/vulkan/internals/VulkanDispatchTableContractTests.cpp
            tests/vulkan/internals/VulkanOptionalFeatureContractTests.cpp
            tests/vulkan/internals/VulkanValidUsageContractTests.cpp
        )
        target_include_directories(RachisRHITests PRIVATE "${rachis_rhi_source_dir}")
        target_compile_definitions(RachisRHITests
            PRIVATE
                RACHIS_ENABLE_D3D12=0
                RACHIS_ENABLE_VULKAN=0
                RACHIS_ENABLE_METAL=1
                RACHIS_TESTING
        )
        target_link_libraries(RachisRHITests PRIVATE Rachis::RHI RachisRHITestSupport RachisRHITestHooks RachisRHIMetalSurfaceTestHost Catch2::Catch2WithMain
            "${RACHIS_APPKIT_FRAMEWORK}" "${RACHIS_QUARTZCORE_FRAMEWORK}" "${RACHIS_METAL_FRAMEWORK}")
        if(TARGET RachisRHITestMetalShaders)
            add_dependencies(RachisRHITests RachisRHITestMetalShaders)
            target_compile_definitions(RachisRHITests PRIVATE
                RACHIS_METALLIB_DIR="${rachis_rhi_metallib_dir}"
                RACHIS_SLANGC_EXECUTABLE="${RACHIS_SLANGC_EXECUTABLE}"
                RACHIS_PARITY_EVIDENCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/parity-evidence"
            )
        endif()
        rachis_configure_rhi_target(RachisRHITests)
        set_target_properties(RachisRHITests PROPERTIES
            MACOSX_RPATH ON
            BUILD_RPATH "@loader_path"
        )
        catch_discover_tests(
            RachisRHITests
            TEST_PREFIX "RachisRHI."
            DISCOVERY_MODE PRE_TEST
            ADD_TAGS_AS_LABELS
            PROPERTIES TIMEOUT 30
        )
    endif()
    if(TARGET RachisRHITests)
        rachis_ensure_shader_binary_tool(rachis_test_shader_binary_tool_cmd rachis_test_shader_binary_tool_depends)
        target_compile_definitions(RachisRHITests PRIVATE
            RACHIS_SHADER_BINARY_TOOL_PATH="${rachis_test_shader_binary_tool_cmd}"
        )
        if(TARGET "${rachis_test_shader_binary_tool_depends}")
            add_dependencies(RachisRHITests "${rachis_test_shader_binary_tool_depends}")
        endif()
        # Shader interfaces that rachis_add_shader generates, used by GeneratedShaderInterfaceTests.cpp on
        # every backend of the build.
        rachis_add_shader(RachisRHITests
            NAME GeneratedComputePush
            SOURCE tests/shaders/CAbiComputePush.hlsl
            ENTRY main
            STAGE compute
            REFLECT)
        rachis_add_shader(RachisRHITests
            NAME GeneratedVertexInput
            SOURCE tests/shaders/VertexInput.hlsl
            ENTRY vertexMain
            STAGE vertex
            REFLECT)
        rachis_add_shader(RachisRHITests
            NAME GeneratedLayouts
            SOURCE tests/shaders/GeneratedLayouts.hlsl
            ENTRY layoutMain
            STAGE compute
            REFLECT)
        rachis_add_shader(RachisRHITests
            NAME GeneratedVertexInputPixel
            SOURCE tests/shaders/VertexInput.hlsl
            ENTRY pixelMain
            STAGE pixel
            REFLECT)
        # Slang sources, and an HLSL source compiled as Slang, cooked with slangc for every backend.
        rachis_add_shader(RachisRHITests
            NAME GeneratedSlang
            SOURCE tests/shaders/GeneratedSlang.slang
            ENTRY slangMain
            STAGE compute
            REFLECT)
        rachis_add_shader(RachisRHITests
            NAME GeneratedLayoutsSlang
            SOURCE tests/shaders/GeneratedLayouts.hlsl
            LANGUAGE SLANG
            ENTRY layoutMain
            STAGE compute
            REFLECT)
        if(RACHIS_ENABLE_VULKAN)
            # GLSL compiles for Vulkan only; its interfaces come from glslang's SPIR-V.
            rachis_add_shader(RachisRHITests
                NAME GeneratedGlslCompute
                SOURCE tests/shaders/GeneratedGlsl.comp
                ENTRY glslMain
                STAGE compute
                REFLECT)
            rachis_add_shader(RachisRHITests
                NAME GeneratedGlslVertex
                SOURCE tests/shaders/GeneratedGlsl.vert
                STAGE vertex
                REFLECT)
            rachis_add_shader(RachisRHITests
                NAME GeneratedGlslPixel
                SOURCE tests/shaders/GeneratedGlsl.frag
                STAGE pixel
                REFLECT)
        endif()
        if(RACHIS_ENABLE_D3D12)
            # Stage pairs for StageLinkageContractTests.cpp.
            rachis_add_shader(RachisRHITests
                NAME StageLinkageVertex
                SOURCE tests/shaders/StageLinkage.hlsl
                ENTRY vertexMain
                STAGE vertex
                REFLECT)
            # StageLinkage<name> compiles the pixel shader entry point pixel<name without "Pixel">.
            foreach(entry IN ITEMS PixelMatching PixelFewer Missing TypeMismatch MoreComponents NoPosition)
                string(REGEX REPLACE "^Pixel" "" stem "${entry}")
                rachis_add_shader(RachisRHITests
                    NAME StageLinkage${entry}
                    SOURCE tests/shaders/StageLinkage.hlsl
                    ENTRY pixel${stem}
                    STAGE pixel
                    REFLECT)
            endforeach()
        endif()
        # Shaders that include Rachis/DrawIndex.hlsli through the include directory rachis_add_shader passes.
        rachis_add_shader(RachisRHITests
            NAME GeneratedDrawIndexVertex
            SOURCE tests/shaders/DrawIndex.hlsl
            ENTRY drawIndexVertex
            STAGE vertex
            REFLECT)
        rachis_add_shader(RachisRHITests
            NAME GeneratedDrawIndexVertexSlang
            SOURCE tests/shaders/DrawIndex.hlsl
            LANGUAGE SLANG
            ENTRY drawIndexVertex
            STAGE vertex
            REFLECT)
        rachis_add_shader(RachisRHITests
            NAME GeneratedDrawIndexPixel
            SOURCE tests/shaders/DrawIndex.hlsl
            ENTRY recordPixel
            STAGE pixel
            REFLECT)
        add_executable(RachisFakeSlangc tests/support/FakeSlangc.cpp)
        rachis_configure_rhi_target(RachisFakeSlangc)
        set_target_properties(RachisFakeSlangc PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shader-tools"
        )
        target_compile_definitions(RachisRHITests PRIVATE
            RACHIS_FAKE_SLANGC_PATH="$<TARGET_FILE:RachisFakeSlangc>"
        )
        add_dependencies(RachisRHITests RachisFakeSlangc)
        rachis_add_shader(RachisRHITests
            NAME PortableReadOnlyStorageShader
            SOURCE tests/shaders/PortableReadOnlyStorage.hlsl
            ENTRY readOnlyMain
            STAGE compute)
        rachis_add_shader(RachisRHITests
            NAME PortableMatrixLayoutShader
            SOURCE tests/shaders/PortableMatrixLayout.hlsl
            ENTRY matrixLayoutMain
            STAGE compute)
        rachis_add_shader(RachisRHITests
            NAME PortableDrawIdVertex
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PortableDrawIds.hlsl"
            ENTRY drawIdVertex
            STAGE vertex)
        rachis_add_shader(RachisRHITests
            NAME PortableDrawIdPixel
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PortableDrawIds.hlsl"
            ENTRY drawIdPixel
            STAGE pixel)
        rachis_add_shader(RachisRHITests
            NAME PortableDrawIdVertexSlang
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PortableDrawIds.hlsl"
            LANGUAGE SLANG
            ENTRY drawIdVertex
            STAGE vertex)
        rachis_add_shader(RachisRHITests
            NAME PortableDrawIdPixelSlang
            SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PortableDrawIds.hlsl"
            LANGUAGE SLANG
            ENTRY drawIdPixel
            STAGE pixel)
        if(TARGET RachisCAbiShaders)
            add_dependencies(RachisRHITests RachisCAbiShaders)
            target_compile_definitions(RachisRHITests PRIVATE
                RACHIS_TEST_PORTABLE_COMPUTE_SHADER_DIR="${rachis_c_abi_shader_dir}"
            )
        endif()
    endif()
endif()

set_target_properties(RachisRHI PROPERTIES EXPORT_NAME RHI DEBUG_POSTFIX d)
set_target_properties(RachisRHIShared PROPERTIES EXPORT_NAME C)
set(rachis_package_targets RachisRHI RachisRHIShared)
if(TARGET RachisShaderBinaryTool)
    set_target_properties(RachisShaderBinaryTool PROPERTIES EXPORT_NAME ShaderBinaryTool)
    list(APPEND rachis_package_targets RachisShaderBinaryTool)
endif()
if(RACHIS_ENABLE_D3D12)
    set_target_properties(RachisRHID3D12 PROPERTIES EXPORT_NAME RHID3D12 DEBUG_POSTFIX d)
    list(APPEND rachis_package_targets RachisRHID3D12)
endif()
if(RACHIS_ENABLE_VULKAN)
    set_target_properties(RachisRHIVulkan PROPERTIES EXPORT_NAME RHIVulkan DEBUG_POSTFIX d)
    list(APPEND rachis_package_targets RachisRHIVulkan)
endif()
if(RACHIS_ENABLE_METAL)
    set_target_properties(RachisRHIMetal PROPERTIES EXPORT_NAME RHIMetal DEBUG_POSTFIX d)
    list(APPEND rachis_package_targets RachisRHIMetal)
endif()

if(RACHIS_INSTALL)
set(rachis_package_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Rachis")
set(rachis_private_library_dir "${CMAKE_INSTALL_LIBDIR}/Rachis")
set(rachis_runtime_dir "${CMAKE_INSTALL_BINDIR}")
set(rachis_include_dir "${CMAKE_INSTALL_INCLUDEDIR}")

function(rachis_install_bundled_archive name target)
    get_target_property(type ${target} TYPE)
    if(NOT type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR
            "The Rachis package bundles ${target} as a static archive, but ${target} is a ${type}. "
            "Set RACHIS_INSTALL=OFF or provide a static ${target}.")
    endif()
    set(config "$<IF:$<BOOL:$<CONFIG>>,$<UPPER_CASE:$<CONFIG>>,NOCONFIG>")
    set(package_file "${CMAKE_CURRENT_BINARY_DIR}/RachisBundled-${name}-$<LOWER_CASE:${config}>.cmake")
    set(link_libraries "")
    if(ARGN)
        set(link_libraries "\n    INTERFACE_LINK_LIBRARIES \"${ARGN}\"")
    endif()
    file(GENERATE OUTPUT "${package_file}" CONTENT
"set_property(TARGET Rachis::${name} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${config})
set_target_properties(Rachis::${name} PROPERTIES
    IMPORTED_LOCATION_${config} \"\${rachis_private_library_dir}/$<TARGET_FILE_NAME:${target}>\"${link_libraries})
")
    install(FILES "$<TARGET_FILE:${target}>" DESTINATION "${rachis_private_library_dir}")
    install(FILES "${package_file}" DESTINATION "${rachis_package_cmake_dir}")
endfunction()

if(RACHIS_ENABLE_D3D12)
    rachis_install_bundled_archive(D3D12MA D3D12MemoryAllocator)
    if(RACHIS_ENABLE_DEBUG_MARKERS)
        install(FILES "$<TARGET_FILE:RachisWinPixEventRuntime>" DESTINATION "${rachis_runtime_dir}")
        install(FILES "$<TARGET_LINKER_FILE:RachisWinPixEventRuntime>" DESTINATION "${rachis_private_library_dir}")
    endif()
endif()
if(RACHIS_ENABLE_VULKAN)
    rachis_install_bundled_archive(volk volk ${CMAKE_DL_LIBS})
endif()

install(TARGETS ${rachis_package_targets} EXPORT RachisTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY include/Rachis DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hlsli")
install(FILES LICENSE THIRD_PARTY_NOTICES.md README.md CHANGELOG.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/Rachis")
install(DIRECTORY docs
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/Rachis"
    FILES_MATCHING PATTERN "*.md")
install(FILES LICENSE THIRD_PARTY_NOTICES.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Rachis")
install(DIRECTORY third_party/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/Rachis/third_party"
    FILES_MATCHING PATTERN "*.txt" PATTERN "*.md")
install(DIRECTORY third_party/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Rachis/third_party"
    FILES_MATCHING PATTERN "*.txt" PATTERN "*.md")
configure_package_config_file(cmake/RachisConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/RachisConfig.cmake"
    INSTALL_DESTINATION "${rachis_package_cmake_dir}"
    PATH_VARS rachis_private_library_dir rachis_runtime_dir rachis_include_dir)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/RachisConfigVersion.cmake"
    VERSION "${RACHIS_PACKAGE_VERSION}" COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/RachisConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/RachisConfigVersion.cmake"
    DESTINATION "${rachis_package_cmake_dir}")
install(EXPORT RachisTargets FILE RachisTargets.cmake NAMESPACE Rachis::
    DESTINATION "${rachis_package_cmake_dir}")
install(FILES
    cmake/RachisShaders.cmake
    cmake/RachisShaderCooking.cmake
    cmake/RachisShaderBinaryTools.cmake
    cmake/WriteShaderArtifactKey.cmake
    cmake/WriteShaderDepfile.cmake
    DESTINATION "${rachis_package_cmake_dir}")
endif()

if(PROJECT_IS_TOP_LEVEL AND rachis_build_tests)
    set(rachis_build_examples_default ON)
else()
    set(rachis_build_examples_default OFF)
endif()
option(RACHIS_BUILD_EXAMPLES "Build the programs in examples/" ${rachis_build_examples_default})
if(RACHIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(rachis_build_tests)
    set(rachis_shared_library_version_arguments
        "-DABI_VERSION=${rachis_c_abi_version}"
        "-DLIBRARY_VERSION=${rachis_c_library_version}"
        "-DREADELF=${CMAKE_READELF}"
        "-DDUMPBIN=${rachis_dumpbin}")
    add_test(NAME RachisCAbiLibraryVersion
        COMMAND "${CMAKE_COMMAND}"
            "-DLIBRARY=$<TARGET_FILE:RachisRHIShared>"
            ${rachis_shared_library_version_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckSharedLibraryVersion.cmake")
    set_tests_properties(RachisCAbiLibraryVersion PROPERTIES LABELS "size:small" TIMEOUT 30)

    set(rachis_project_test_arguments
        "-DCONFIG=$<CONFIG>"
        "-DGENERATOR=${CMAKE_GENERATOR}"
        "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DC_COMPILER=${CMAKE_C_COMPILER}"
        "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DRC_COMPILER=${CMAKE_RC_COMPILER}"
        "-DMT_TOOL=${CMAKE_MT}"
        "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        "-DOSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
        "-DOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
        "-DSYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
        "-DSYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    set(rachis_shader_test_arguments
        "-DDXC=${RACHIS_DXC_EXECUTABLE}"
        "-DDXC_SPIRV=${RACHIS_DXC_SPIRV_EXECUTABLE}"
        "-DSPIRV_VAL=${RACHIS_SPIRV_VAL_EXECUTABLE}"
        "-DSLANGC=${RACHIS_SLANGC_EXECUTABLE}"
        "-DSHADER_BINARY_TOOL=${RACHIS_SHADER_BINARY_TOOL}")
    if(RACHIS_ENABLE_METAL)
        set(rachis_without_slangc_argument -DWITHOUT_SLANGC=OFF)
    else()
        set(rachis_without_slangc_argument -DWITHOUT_SLANGC=ON)
    endif()

    set(rachis_notice_dependencies)
    set(rachis_fetched_source_arguments)
    foreach(rachis_dependency IN LISTS rachis_fetched_dependencies)
        string(TOLOWER "${rachis_dependency}" rachis_dependency_lower)
        string(TOUPPER "${rachis_dependency}" rachis_dependency_upper)
        set(rachis_dependency_source_dir "${${rachis_dependency_lower}_SOURCE_DIR}")
        list(APPEND rachis_notice_dependencies "-DFETCHED_${rachis_dependency}=${rachis_dependency_source_dir}")
        if(rachis_dependency_source_dir)
            list(APPEND rachis_fetched_source_arguments
                "-DFETCHCONTENT_SOURCE_DIR_${rachis_dependency_upper}=${rachis_dependency_source_dir}")
        endif()
    endforeach()
    add_test(NAME RachisThirdPartyNotices
        COMMAND "${CMAKE_COMMAND}"
            "-DNOTICES=${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
            ${rachis_notice_dependencies}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckThirdPartyNotices.cmake")
    set_tests_properties(RachisThirdPartyNotices PROPERTIES LABELS "size:small" TIMEOUT 30)

    if(UNIX)
        find_program(rachis_bash bash NO_CACHE)
        find_program(rachis_perl perl NO_CACHE)
        if(rachis_bash AND rachis_perl)
            add_test(NAME RachisTlcTimeout
                COMMAND "${CMAKE_COMMAND}"
                    "-DSCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tools/tlc.sh"
                    "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rtlc/$<CONFIG>"
                    "-DBASH=${rachis_bash}"
                    "-DPERL=${rachis_perl}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckTlcTimeout.cmake")
            set_tests_properties(RachisTlcTimeout PROPERTIES LABELS "size:small" TIMEOUT 120)
        endif()
    endif()

    if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
        set(rachis_ninja_file "build-$<CONFIG>.ninja")
    else()
        set(rachis_ninja_file "build.ninja")
    endif()
    if(TARGET RachisWinPixEventRuntimeCopy AND CMAKE_GENERATOR MATCHES "^Ninja")
        get_directory_property(rachis_runtime_dll_targets BUILDSYSTEM_TARGETS)
        if(RACHIS_BUILD_EXAMPLES)
            get_directory_property(rachis_example_targets DIRECTORY examples BUILDSYSTEM_TARGETS)
            list(APPEND rachis_runtime_dll_targets ${rachis_example_targets})
        endif()
        set(rachis_runtime_dll_directories)
        foreach(rachis_target IN LISTS rachis_runtime_dll_targets)
            get_target_property(rachis_target_type ${rachis_target} TYPE)
            get_target_property(rachis_target_links ${rachis_target} LINK_LIBRARIES)
            if(rachis_target_type MATCHES "^(EXECUTABLE|SHARED_LIBRARY)$"
                    AND "${rachis_target_links}" MATCHES "(^|;)Rachis::(RHI|C)(;|$)")
                list(APPEND rachis_runtime_dll_directories "$<TARGET_FILE_DIR:${rachis_target}>")
            endif()
        endforeach()
        list(JOIN rachis_runtime_dll_directories "|" rachis_runtime_dll_directories)
        add_test(NAME RachisRuntimeDllCopies
            COMMAND "${CMAKE_COMMAND}"
                "-DNINJA=${CMAKE_MAKE_PROGRAM}"
                "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
                "-DNINJA_FILE=${rachis_ninja_file}"
                "-DFILE_NAME=$<TARGET_FILE_NAME:RachisWinPixEventRuntime>"
                "-DDIRECTORIES=${rachis_runtime_dll_directories}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckRuntimeDllCopies.cmake")
        set_tests_properties(RachisRuntimeDllCopies PROPERTIES LABELS "size:small" TIMEOUT 30)
    endif()

    if(rachis_rhi_test_dxil_outputs AND CMAKE_GENERATOR MATCHES "^Ninja")
        list(JOIN rachis_rhi_test_dxil_outputs "|" rachis_test_shader_outputs)
        list(JOIN rachis_rhi_test_dxil_sources "|" rachis_test_shader_sources)
        add_test(NAME RachisTestShaderRules
            COMMAND "${CMAKE_COMMAND}"
                "-DNINJA=${CMAKE_MAKE_PROGRAM}"
                "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
                "-DNINJA_FILE=${rachis_ninja_file}"
                "-DOUTPUTS=${rachis_test_shader_outputs}"
                "-DSOURCES=${rachis_test_shader_sources}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckShaderRules.cmake")
        set_tests_properties(RachisTestShaderRules PROPERTIES LABELS "size:small" TIMEOUT 30)
    endif()

    if(RACHIS_ENABLE_VULKAN)
        set(rachis_mipmaps_spirv_keys
            "${rachis_rhi_spirv_dir}/GenerateMipmaps1D.spv.key"
            "${rachis_rhi_spirv_dir}/GenerateMipmaps2D.spv.key"
            "${rachis_rhi_spirv_dir}/GenerateMipmaps3D.spv.key")
        list(JOIN rachis_mipmaps_spirv_keys "|" rachis_mipmaps_spirv_keys)
        list(JOIN RACHIS_SHADER_DXC_SPIRV_OPTIONS " " rachis_mipmaps_spirv_option_text)
        list(JOIN rachis_mipmaps_spirv_options " " rachis_mipmaps_spirv_extra_option_text)
        add_test(NAME RachisShaderArtifactKeys
            COMMAND "${CMAKE_COMMAND}"
                "-DKEYS=${rachis_mipmaps_spirv_keys}"
                "-DOPTIONS=${rachis_mipmaps_spirv_option_text} ${rachis_mipmaps_spirv_extra_option_text}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckShaderArtifactKeys.cmake")
        set_tests_properties(RachisShaderArtifactKeys PROPERTIES LABELS "size:small" TIMEOUT 30)
    endif()

    add_test(NAME RachisEmbeddedProject
        COMMAND "${CMAKE_COMMAND}"
            "-DRACHIS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/subdirectories"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rembed/$<CONFIG>"
            ${rachis_project_test_arguments}
            ${rachis_shader_test_arguments}
            "-DBACKENDS=${RACHIS_BACKENDS}"
            "-DDEBUG_MARKERS=${RACHIS_ENABLE_DEBUG_MARKERS}"
            ${rachis_without_slangc_argument}
            ${rachis_fetched_source_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckEmbeddedProject.cmake")
    set_tests_properties(RachisEmbeddedProject PROPERTIES LABELS "size:small" TIMEOUT 300)
endif()

if(rachis_build_tests AND RACHIS_INSTALL)
    if(RACHIS_ENABLE_D3D12)
        set(rachis_package_consumer_size medium)
    else()
        set(rachis_package_consumer_size large)
    endif()
    add_test(NAME RachisPackageConsumer
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/consumer"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg/$<CONFIG>"
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            "-DBINDIR=${CMAKE_INSTALL_BINDIR}"
            ${rachis_shared_library_version_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageConsumer PROPERTIES
        LABELS "size:${rachis_package_consumer_size}"
        TIMEOUT 300)

    # Builds examples/ against the installed package, the way a project outside this tree would,
    # including rachis_add_shader, and runs the examples.
    add_test(NAME RachisPackageExamples
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/examples"
            "-DWORK_DIR=${CMAKE_BINARY_DIR}/rpkg-examples/$<CONFIG>"
            "-DVARIANTS=examples"
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            ${rachis_shader_test_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageExamples PROPERTIES
        LABELS "size:${rachis_package_consumer_size}"
        TIMEOUT 600)

    add_test(NAME RachisPackageSiblings
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/subdirectories"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg-siblings/$<CONFIG>"
            "-DVARIANTS=siblings"
            -DCONFIGURE_ONLY=ON
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            ${rachis_shader_test_arguments}
            ${rachis_without_slangc_argument}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageSiblings PROPERTIES LABELS "size:small" TIMEOUT 300)

    add_test(NAME RachisPackageCrossCompiling
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/subdirectories"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg-cross/$<CONFIG>"
            "-DVARIANTS=siblings"
            -DCONFIGURE_ONLY=ON
            -DCROSS_COMPILING=ON
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            "-DBINDIR=${CMAKE_INSTALL_BINDIR}"
            ${rachis_shader_test_arguments}
            ${rachis_without_slangc_argument}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageCrossCompiling PROPERTIES LABELS "size:small" TIMEOUT 300)

    add_test(NAME RachisPackageConfig
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/config"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg-config/$<CONFIG>"
            "-DVARIANTS=config"
            -DCONFIGURE_ONLY=ON
            -DADD_RELEASE_EXPORT=ON
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageConfig PROPERTIES LABELS "size:small" TIMEOUT 300)

    add_test(NAME RachisPackageShaderNames
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/shadernames"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg-shadernames/$<CONFIG>"
            "-DVARIANTS=shadernames"
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            ${rachis_shader_test_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageShaderNames PROPERTIES LABELS "size:small" TIMEOUT 300)

    add_test(NAME RachisPackageShaderIncludes
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/includes"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg-includes/$<CONFIG>"
            "-DVARIANTS=includes"
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            ${rachis_shader_test_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageShaderIncludes PROPERTIES LABELS "size:small" TIMEOUT 300)

    add_test(NAME RachisPackageShaderBuildPaths
        COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package/buildpaths"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/rpkg-buildpaths/$<CONFIG>"
            "-DVARIANTS=buildpaths"
            "-DBUILD_NAME=build,comma"
            ${rachis_project_test_arguments}
            "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
            ${rachis_shader_test_arguments}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/CheckPackageConsumer.cmake")
    set_tests_properties(RachisPackageShaderBuildPaths PROPERTIES LABELS "size:small" TIMEOUT 300)
endif()
