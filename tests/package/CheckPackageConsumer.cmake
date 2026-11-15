set(required_variables BUILD_DIR SOURCE_DIR WORK_DIR CONFIG GENERATOR LIBDIR CTEST_COMMAND)
if(CROSS_COMPILING)
    list(APPEND required_variables BINDIR SYSTEM_NAME SYSTEM_PROCESSOR)
endif()
foreach(required IN LISTS required_variables)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required.")
    endif()
endforeach()

function(rachis_package_run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        list(JOIN ARGN " " command)
        message(FATAL_ERROR "Package consumer step failed (${status}): ${command}")
    endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
rachis_package_run("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --config "${CONFIG}" --prefix "${prefix}")

foreach(directory IN ITEMS LIBDIR BINDIR)
    if(IS_ABSOLUTE "${${directory}}")
        set(installed_${directory} "${${directory}}")
    else()
        set(installed_${directory} "${prefix}/${${directory}}")
    endif()
endforeach()

string(TOLOWER "${CONFIG}" config_lower)
if(ADD_RELEASE_EXPORT AND NOT config_lower STREQUAL "release")
    string(TOUPPER "${CONFIG}" config_upper)
    file(GLOB exports "${installed_LIBDIR}/cmake/Rachis/*-${config_lower}.cmake")
    if(NOT exports)
        message(FATAL_ERROR "The install wrote no ${CONFIG} export files to ${installed_LIBDIR}/cmake/Rachis.")
    endif()
    foreach(export IN LISTS exports)
        file(READ "${export}" content)
        string(REGEX REPLACE "([^A-Za-z0-9])${config_upper}([^A-Za-z0-9])" "\\1RELEASE\\2" content "${content}")
        string(REGEX REPLACE "-${config_lower}\\.cmake$" "-release.cmake" release_export "${export}")
        file(WRITE "${release_export}" "${content}")
    endforeach()
endif()

foreach(leaked IN ITEMS
        "${prefix}/include/D3D12MemAlloc.h"
        "${prefix}/share/cmake/D3D12MemoryAllocator"
        "${prefix}/lib/D3D12MA.lib"
        "${prefix}/lib/D3D12MAd.lib"
        "${prefix}/include/volk.h"
        "${prefix}/include/vk_mem_alloc.h"
        "${prefix}/include/vulkan")
    if(EXISTS "${leaked}")
        message(FATAL_ERROR "The Rachis package installed a bundled dependency's public file: ${leaked}")
    endif()
endforeach()

if(DEFINED ABI_VERSION)
    if(CMAKE_HOST_WIN32)
        set(installed_c_library "${installed_BINDIR}/rachis.dll")
    elseif(CMAKE_HOST_APPLE)
        set(installed_c_library "${installed_LIBDIR}/librachis.dylib")
    else()
        set(installed_c_library "${installed_LIBDIR}/librachis.so")
    endif()
    rachis_package_run("${CMAKE_COMMAND}"
        "-DLIBRARY=${installed_c_library}"
        "-DABI_VERSION=${ABI_VERSION}"
        "-DLIBRARY_VERSION=${LIBRARY_VERSION}"
        "-DREADELF=${READELF}"
        "-DDUMPBIN=${DUMPBIN}"
        -P "${CMAKE_CURRENT_LIST_DIR}/CheckSharedLibraryVersion.cmake")
endif()

set(decoy_prefix "${WORK_DIR}/decoy")
file(WRITE "${decoy_prefix}/share/cmake/D3D12MemoryAllocator/D3D12MemoryAllocatorConfig.cmake"
    "message(FATAL_ERROR \"The Rachis package loaded a D3D12MemoryAllocator package from CMAKE_PREFIX_PATH.\")")

set(common_arguments -G "${GENERATOR}"
    "-DRachis_DIR=${installed_LIBDIR}/cmake/Rachis"
    "-DCMAKE_PREFIX_PATH=${decoy_prefix}"
    "-DCMAKE_BUILD_TYPE=${CONFIG}")
if(MAKE_PROGRAM)
    list(APPEND common_arguments "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(C_COMPILER)
    list(APPEND common_arguments "-DCMAKE_C_COMPILER=${C_COMPILER}")
endif()
if(RC_COMPILER)
    list(APPEND common_arguments "-DCMAKE_RC_COMPILER=${RC_COMPILER}")
endif()
if(MT_TOOL)
    list(APPEND common_arguments "-DCMAKE_MT=${MT_TOOL}")
endif()
if(OSX_ARCHITECTURES)
    list(APPEND common_arguments "-DCMAKE_OSX_ARCHITECTURES=${OSX_ARCHITECTURES}")
endif()
if(OSX_DEPLOYMENT_TARGET)
    list(APPEND common_arguments "-DCMAKE_OSX_DEPLOYMENT_TARGET=${OSX_DEPLOYMENT_TARGET}")
endif()
# The shader compilers this build found, so rachis_add_shader in the consumer finds the same ones.
set(tools DXC DXC_SPIRV SPIRV_VAL)
set(environment)
if(WITHOUT_SLANGC)
    list(APPEND common_arguments -DRACHIS_PACKAGE_WITHOUT_SLANGC=ON)
    set(environment "${CMAKE_COMMAND}" -E env --unset=SLANGC)
else()
    list(APPEND tools SLANGC)
endif()
foreach(tool IN LISTS tools)
    if(${tool})
        list(APPEND common_arguments "-DRACHIS_${tool}_EXECUTABLE=${${tool}}")
    endif()
endforeach()
if(SHADER_BINARY_TOOL)
    list(APPEND common_arguments "-DRACHIS_SHADER_BINARY_TOOL=${SHADER_BINARY_TOOL}")
endif()

# cpp and c build tests/package/consumer as C++ and as C only; any other variant is built as is.
if(NOT VARIANTS)
    set(VARIANTS cpp c)
endif()
if(NOT BUILD_NAME)
    set(BUILD_NAME build)
endif()
foreach(variant IN LISTS VARIANTS)
    set(source "${WORK_DIR}/${variant}/source")
    set(build "${WORK_DIR}/${variant}/${BUILD_NAME}")
    file(COPY "${SOURCE_DIR}/" DESTINATION "${source}")
    set(arguments ${common_arguments})
    if(variant STREQUAL "c")
        list(APPEND arguments -DRACHIS_PACKAGE_C_ONLY=ON)
    elseif(CXX_COMPILER)
        list(APPEND arguments "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
    endif()
    if(CROSS_COMPILING)
        list(FILTER arguments EXCLUDE REGEX "^-DRACHIS_SHADER_BINARY_TOOL=")
        list(APPEND arguments "-DCMAKE_SYSTEM_NAME=${SYSTEM_NAME}" "-DCMAKE_SYSTEM_PROCESSOR=${SYSTEM_PROCESSOR}")
        execute_process(
            COMMAND ${environment} "${CMAKE_COMMAND}" -S "${source}" -B "${build}-without-tool" ${arguments}
            RESULT_VARIABLE status
            OUTPUT_VARIABLE output
            ERROR_VARIABLE output)
        string(REGEX REPLACE "[ \t\r\n]+" " " flat_output "${output}")
        if(status EQUAL 0)
            message(FATAL_ERROR
                "${variant} configured for cross-compiling without RACHIS_SHADER_BINARY_TOOL, so its build "
                "would run a RachisShaderBinaryTool that cannot run on the host.")
        endif()
        if(NOT flat_output MATCHES "Set RACHIS_SHADER_BINARY_TOOL to a host build of it when cross-compiling")
            message(FATAL_ERROR
                "${variant} failed to configure for cross-compiling, but not because "
                "RACHIS_SHADER_BINARY_TOOL is missing:\n${output}")
        endif()
        if(SHADER_BINARY_TOOL)
            set(host_tool "${SHADER_BINARY_TOOL}")
        elseif(CMAKE_HOST_WIN32)
            set(host_tool "${installed_BINDIR}/RachisShaderBinaryTool.exe")
        else()
            set(host_tool "${installed_BINDIR}/RachisShaderBinaryTool")
        endif()
        list(APPEND arguments "-DRACHIS_SHADER_BINARY_TOOL=${host_tool}")
    endif()
    rachis_package_run(${environment} "${CMAKE_COMMAND}" -S "${source}" -B "${build}" ${arguments})
    if(CONFIGURE_ONLY)
        continue()
    endif()
    rachis_package_run("${CMAKE_COMMAND}" --build "${build}" --config "${CONFIG}")
    rachis_package_run("${CTEST_COMMAND}" --test-dir "${build}" -C "${CONFIG}" --output-on-failure --no-tests=error)
endforeach()
