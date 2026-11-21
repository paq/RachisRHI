cmake_minimum_required(VERSION 4.0)
project(RachisExamples LANGUAGES CXX)

# Inside the RachisRHI source tree the targets already exist. Anywhere else, add the install prefix
# to CMAKE_PREFIX_PATH, or point Rachis_DIR at <libdir>/cmake/Rachis under it.
if(NOT TARGET Rachis::RHI)
    find_package(Rachis 0.3 CONFIG REQUIRED)
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
endif()

set(rachis_example_backends)
foreach(backend IN ITEMS D3D12 VULKAN METAL)
    get_target_property(enabled Rachis::RHI RACHIS_ENABLE_${backend})
    if(enabled)
        string(TOLOWER "${backend}" backend_name)
        list(APPEND rachis_example_backends "${backend_name}")
    endif()
endforeach()

function(rachis_add_example name)
    add_executable(rachis-example-${name} ${name}/main.cpp)
    target_compile_features(rachis-example-${name} PRIVATE cxx_std_23)
    target_include_directories(rachis-example-${name} PRIVATE common)
    target_link_libraries(rachis-example-${name} PRIVATE Rachis::RHI)
    if(WIN32 AND TARGET RachisWinPixEventRuntime)
        # Debug markers load WinPixEventRuntime.dll, which must sit beside the executable.
        if(NOT TARGET rachis-example-pix-runtime)
            add_custom_target(rachis-example-pix-runtime
                COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:rachis-example-${name}>"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "$<TARGET_FILE:RachisWinPixEventRuntime>" "$<TARGET_FILE_DIR:rachis-example-${name}>"
                COMMENT "Copying WinPixEventRuntime.dll beside the examples"
                VERBATIM)
        endif()
        add_dependencies(rachis-example-${name} rachis-example-pix-runtime)
    endif()

    # One test per backend. Direct3D 12 runs on WARP so the test needs no GPU; exit code 77 means
    # the backend cannot run on this machine.
    foreach(backend IN LISTS rachis_example_backends)
        set(arguments --backend ${backend})
        set(size medium)
        if(backend STREQUAL "d3d12")
            list(APPEND arguments --warp)
        elseif(backend STREQUAL "metal")
            set(size large)
        endif()
        if(name STREQUAL "triangle")
            list(APPEND arguments --output "${CMAKE_CURRENT_BINARY_DIR}/triangle-${backend}.ppm")
        endif()
        set(labels "size:${size}" example)
        if(backend STREQUAL "vulkan")
            list(APPEND labels "requires:HardwareVulkan")
        endif()
        add_test(NAME RachisExample.${name}.${backend} COMMAND rachis-example-${name} ${arguments})
        set_tests_properties(RachisExample.${name}.${backend} PROPERTIES
            LABELS "${labels}"
            SKIP_RETURN_CODE 77
            TIMEOUT 60)
    endforeach()
endfunction()

rachis_add_example(compute)
rachis_add_shader(rachis-example-compute
    NAME SquareShader
    SOURCE compute/Square.hlsl
    ENTRY squareMain
    STAGE compute
    REFLECT)

rachis_add_example(triangle)
rachis_add_shader(rachis-example-triangle
    NAME TriangleVertex
    SOURCE triangle/Triangle.hlsl
    ENTRY vertexMain
    STAGE vertex
    REFLECT)
rachis_add_shader(rachis-example-triangle
    NAME TrianglePixel
    SOURCE triangle/Triangle.hlsl
    ENTRY pixelMain
    STAGE pixel
    REFLECT)
