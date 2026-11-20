# Runs WriteShaderReflection.cmake and WriteShaderDepfile.cmake on dependency lists that name a file whose
# name contains a semicolon and a space, and checks that each depfile names that file once, whole. Then runs
# WriteShaderArtifactKey.cmake on each depfile and checks that the artifact key hashes that file.
foreach(required IN ITEMS SOURCE_DIR WORK_DIR)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required.")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(source "${WORK_DIR}/shader.hlsl")
set(include "${WORK_DIR}/material;layout values.hlsli")
file(WRITE "${source}" "#include \"material;layout values.hlsli\"\n")
file(WRITE "${include}" "static const uint value = 1;\n")
file(SHA256 "${include}" include_hash)
string(REPLACE " " "\\ " escaped_source "${source}")
string(REPLACE " " "\\ " escaped_include "${include}")

set(failures)
function(rachis_check_depfile label depfile)
    file(READ "${depfile}" text)
    string(FIND "${text}" " \\\n  ${escaped_include} \\\n" whole)
    string(FIND "${text}" "\n  ${escaped_include}\n" last)
    if(whole EQUAL -1 AND last EQUAL -1)
        set(failures "${failures}${label} does not name '${include}' whole:\n${text}\n" PARENT_SCOPE)
    endif()
    string(FIND "${text}" "\n  ${escaped_source}" found_source)
    if(found_source EQUAL -1)
        set(failures "${failures}${label} does not name '${source}':\n${text}\n" PARENT_SCOPE)
    endif()
endfunction()

file(WRITE "${WORK_DIR}/fake-slangc.cmake"
    "message(NOTICE \"(0): note: include '${source}'\")\n"
    "message(NOTICE \"(0): note: include   '${include}'\")\n"
    "message(NOTICE \"shader.hlsl(1): warning 1: keep a;b\")\n")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DTARGET=${WORK_DIR}/shader.reflection.json"
        "-DDEPFILE=${WORK_DIR}/shader.reflection.json.d"
        -P "${SOURCE_DIR}/cmake/WriteShaderReflection.cmake"
        --
        "${CMAKE_COMMAND}" -P "${WORK_DIR}/fake-slangc.cmake"
    RESULT_VARIABLE status
    ERROR_VARIABLE diagnostics)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "WriteShaderReflection.cmake failed (${status}):\n${diagnostics}")
endif()
if(NOT diagnostics MATCHES "warning 1: keep a;b")
    string(APPEND failures "WriteShaderReflection.cmake did not pass the slangc warning through:\n${diagnostics}\n")
endif()
rachis_check_depfile("The reflection depfile" "${WORK_DIR}/shader.reflection.json.d")

file(WRITE "${WORK_DIR}/shader.spv.slangc.d" "shader.spv: ${escaped_source} ${escaped_include}\n")
file(WRITE "${WORK_DIR}/shader.dxil.dxc.d" "shader.dxil: ${source} \\\n  ${include}\n")
foreach(format IN ITEMS slangc dxc)
    if(format STREQUAL "slangc")
        set(output "${WORK_DIR}/shader.spv")
    else()
        set(output "${WORK_DIR}/shader.dxil")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DFORMAT=${format}"
            "-DINPUT=${output}.${format}.d"
            "-DOUTPUT=${output}.d"
            "-DTARGET=${output}"
            -P "${SOURCE_DIR}/cmake/WriteShaderDepfile.cmake"
        RESULT_VARIABLE status
        ERROR_VARIABLE diagnostics)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "WriteShaderDepfile.cmake failed on the ${format} depfile (${status}):\n${diagnostics}")
    endif()
    rachis_check_depfile("The ${format} depfile" "${output}.d")

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DOUTPUT=${output}.key"
            "-DSOURCE=${source}"
            "-DDEPFILE=${output}.d"
            "-DCOMPILER=compiler"
            "-DTARGET=${format}"
            "-DPROFILE=profile"
            "-DENTRY=main"
            "-DENVIRONMENT=environment"
            "-DOPTIONS=options"
            "-DDEFINES=defines"
            -P "${SOURCE_DIR}/cmake/WriteShaderArtifactKey.cmake"
        RESULT_VARIABLE status
        ERROR_VARIABLE diagnostics)
    if(NOT status EQUAL 0)
        string(APPEND failures
            "WriteShaderArtifactKey.cmake failed on the ${format} depfile (${status}):\n${diagnostics}\n")
        continue()
    endif()
    file(READ "${output}.key" key)
    string(FIND "${key}" "\ninclude=${include}\ninclude_hash=${include_hash}\n" hashed)
    if(hashed EQUAL -1)
        string(APPEND failures "The ${format} artifact key does not hash '${include}':\n${key}\n")
    endif()
endforeach()

if(failures)
    message(FATAL_ERROR "${failures}")
endif()
message(STATUS "The depfile and artifact-key scripts keep file names with semicolons whole.")
