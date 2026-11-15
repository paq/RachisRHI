cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED NOTICES OR NOT EXISTS "${NOTICES}")
    message(FATAL_ERROR "CheckThirdPartyNotices requires -DNOTICES=<path to THIRD_PARTY_NOTICES.md> (got '${NOTICES}').")
endif()

set(upstream_files_RachisVulkanHeaders
    Apache-2.0-LICENSE.txt LICENSES/Apache-2.0.txt
    Vulkan-Headers-LICENSE.md LICENSE.md)
set(upstream_files_RachisVolk
    volk-LICENSE.md LICENSE.md)
set(upstream_files_RachisVma
    VulkanMemoryAllocator-LICENSE.txt LICENSE.txt)
set(upstream_files_D3D12MemoryAllocator
    D3D12MemoryAllocator-LICENSE.txt LICENSE.txt
    D3D12MemoryAllocator-NOTICES.txt NOTICES.txt)
set(upstream_files_winpixeventruntime
    WinPixEventRuntime-LICENSE.txt license.txt
    WinPixEventRuntime-ThirdPartyNotices.txt ThirdPartyNotices.txt)

get_filename_component(notice_root "${NOTICES}" DIRECTORY)
file(READ "${NOTICES}" notice_text)
string(REPLACE ";" " " notice_text "${notice_text}")
string(REPLACE "[" "<" notice_text "${notice_text}")
string(REPLACE "]" ">" notice_text "${notice_text}")

set(failures)

string(REGEX MATCHALL ">\\([^)\n]+\\)" notice_links "${notice_text}")
set(local_links)
foreach(link IN LISTS notice_links)
    string(REGEX REPLACE "^>\\(([^)]+)\\)$" "\\1" target "${link}")
    if(target MATCHES "^[A-Za-z][A-Za-z0-9+.-]*:")
        continue()
    endif()
    list(APPEND local_links "${target}")
    if(NOT EXISTS "${notice_root}/${target}")
        list(APPEND failures "THIRD_PARTY_NOTICES.md links to ${target}, which does not exist.")
    endif()
endforeach()

file(GLOB license_files RELATIVE "${notice_root}"
    "${notice_root}/third_party/*.txt"
    "${notice_root}/third_party/*.md")
if(NOT license_files)
    list(APPEND failures "No license files found under ${notice_root}/third_party.")
endif()

set(license_names)
foreach(license_file IN LISTS license_files)
    get_filename_component(name "${license_file}" NAME)
    list(APPEND license_names "${name}")
    if(NOT license_file IN_LIST local_links)
        list(APPEND failures "${license_file} is installed with the package, but THIRD_PARTY_NOTICES.md does not link to it.")
    endif()
endforeach()

get_cmake_property(defined_variables VARIABLES)
set(compared_upstreams 0)
foreach(variable IN LISTS defined_variables)
    if(NOT variable MATCHES "^FETCHED_(.+)$")
        continue()
    endif()
    set(dependency "${CMAKE_MATCH_1}")
    set(source_dir "${${variable}}")
    if(NOT DEFINED upstream_files_${dependency})
        list(APPEND failures
            "The build fetched ${dependency}, but this check does not know which third_party files come from it.")
        continue()
    endif()
    set(upstream_files ${upstream_files_${dependency}})
    while(upstream_files)
        list(POP_FRONT upstream_files name upstream_path)
        if(NOT name IN_LIST license_names)
            list(APPEND failures "${dependency} should provide third_party/${name}, which does not exist.")
            continue()
        endif()
        if(NOT source_dir)
            list(APPEND failures
                "The build fetched ${dependency} but passed no source directory, so third_party/${name} was not compared.")
            continue()
        endif()
        set(upstream "${source_dir}/${upstream_path}")
        if(NOT EXISTS "${upstream}")
            list(APPEND failures "The upstream copy of third_party/${name} is missing: ${upstream}")
            continue()
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E compare_files --ignore-eol
                "${notice_root}/third_party/${name}" "${upstream}"
            RESULT_VARIABLE compare_result)
        if(NOT compare_result EQUAL 0)
            list(APPEND failures "third_party/${name} is not a verbatim copy of ${upstream}.")
            continue()
        endif()
        math(EXPR compared_upstreams "${compared_upstreams} + 1")
        message(STATUS "third_party/${name} matches ${dependency}: ${upstream}")
    endwhile()
endforeach()

if(failures)
    list(JOIN failures "\n  " failure_text)
    message(FATAL_ERROR "Third-party notices are inconsistent:\n  ${failure_text}")
endif()

list(LENGTH license_files license_count)
message(STATUS "Checked ${license_count} third-party license files; ${compared_upstreams} compared with their fetched upstream copies.")
