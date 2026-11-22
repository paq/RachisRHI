cmake_minimum_required(VERSION 4.0)
project(RachisPackageConfig LANGUAGES C CXX)

find_package(Rachis 0.4 CONFIG REQUIRED COMPONENTS RHI C)

foreach(component IN ITEMS RHI C RHID3D12 RHIVulkan RHIMetal ShaderBinaryTool)
    if(TARGET Rachis::${component})
        if(NOT Rachis_${component}_FOUND)
            message(FATAL_ERROR
                "The package has Rachis::${component}, but Rachis_${component}_FOUND is "
                "'${Rachis_${component}_FOUND}'.")
        endif()
        find_package(Rachis 0.4 CONFIG QUIET COMPONENTS ${component})
        if(NOT Rachis_FOUND)
            message(FATAL_ERROR
                "find_package(Rachis COMPONENTS ${component}) failed, although the package has "
                "Rachis::${component}.")
        endif()
    else()
        if(NOT DEFINED Rachis_${component}_FOUND OR Rachis_${component}_FOUND)
            message(FATAL_ERROR
                "The package lacks Rachis::${component}, but Rachis_${component}_FOUND is "
                "'${Rachis_${component}_FOUND}'.")
        endif()
        find_package(Rachis 0.4 CONFIG QUIET COMPONENTS ${component})
        if(Rachis_FOUND)
            message(FATAL_ERROR
                "find_package(Rachis COMPONENTS ${component}) succeeded, although the package lacks "
                "Rachis::${component}.")
        endif()
        find_package(Rachis 0.4 CONFIG QUIET OPTIONAL_COMPONENTS ${component})
        if(NOT Rachis_FOUND)
            message(FATAL_ERROR "find_package(Rachis OPTIONAL_COMPONENTS ${component}) failed.")
        endif()
    endif()
endforeach()

find_package(Rachis 0.4 CONFIG QUIET COMPONENTS NotARachisComponent)
if(Rachis_FOUND)
    message(FATAL_ERROR "find_package(Rachis COMPONENTS NotARachisComponent) succeeded.")
endif()

set(rachis_release_targets)
foreach(name IN ITEMS RHI C RHID3D12 RHIVulkan RHIMetal ShaderBinaryTool volk D3D12MA)
    if(NOT TARGET Rachis::${name})
        continue()
    endif()
    get_target_property(configurations Rachis::${name} IMPORTED_CONFIGURATIONS)
    if(NOT "RELEASE" IN_LIST configurations)
        message(FATAL_ERROR "Rachis::${name} has no Release configuration (it has '${configurations}').")
    endif()
    foreach(configuration IN ITEMS RELWITHDEBINFO MINSIZEREL)
        get_target_property(mapping Rachis::${name} MAP_IMPORTED_CONFIG_${configuration})
        if(NOT mapping STREQUAL "RELEASE")
            message(FATAL_ERROR
                "Rachis::${name} maps ${configuration} to '${mapping}' instead of RELEASE, so a "
                "${configuration} application would not use the package's Release build of it.")
        endif()
    endforeach()
    list(APPEND rachis_release_targets Rachis::${name})
endforeach()
list(JOIN rachis_release_targets ", " rachis_release_targets)
message(STATUS
    "Components match the package's targets, and RelWithDebInfo and MinSizeRel use Release for "
    "${rachis_release_targets}.")
