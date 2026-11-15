foreach(required IN ITEMS SCRIPT WORK_DIR BASH PERL)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required.")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(tool_dir "${WORK_DIR}/bin")
set(spec_dir "${WORK_DIR}/spec")
set(perl_dir "${WORK_DIR}/perl")
file(MAKE_DIRECTORY "${tool_dir}" "${spec_dir}" "${perl_dir}" "${WORK_DIR}/tmp")
foreach(tool IN ITEMS basename cat dirname grep head mktemp rm sed sleep tail tr)
    unset(tool_path)
    find_program(tool_path ${tool} NO_CACHE REQUIRED)
    file(CREATE_LINK "${tool_path}" "${tool_dir}/${tool}" SYMBOLIC)
endforeach()
file(CREATE_LINK "${PERL}" "${tool_dir}/perl" SYMBOLIC)

file(WRITE "${WORK_DIR}/tla2tools.jar" "")
file(WRITE "${WORK_DIR}/java" [=[#!/bin/sh
if [ "$1" = -version ]; then
  exit 0
fi
exec sleep 30
]=])
file(CHMOD "${WORK_DIR}/java" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
file(WRITE "${spec_dir}/Hang.tla" "")
file(WRITE "${spec_dir}/Hang.cfg" [=[\* Expect: pass
]=])
file(WRITE "${perl_dir}/RachisDelayForkedChild.pm" [=[package RachisDelayForkedChild;
*CORE::GLOBAL::fork = sub {
    my $pid = CORE::fork();
    select(undef, undef, undef, 3) if defined $pid && $pid == 0;
    return $pid;
};
1;
]=])

unset(ENV{SOURCE_DATE_EPOCH})
string(TIMESTAMP started "%s" UTC)
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${tool_dir}"
        "TMPDIR=${WORK_DIR}/tmp"
        "TLA2TOOLS=${WORK_DIR}/tla2tools.jar"
        "JAVA=${WORK_DIR}/java"
        TLC_TIMEOUT=1
        TLC_WORKERS=1
        "PERL5LIB=${perl_dir}"
        PERL5OPT=-MRachisDelayForkedChild
        "${BASH}" "${SCRIPT}" "${spec_dir}/Hang.cfg"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    TIMEOUT 90)
string(TIMESTAMP finished "%s" UTC)
math(EXPR elapsed "${finished} - ${started}")

set(failures)
if(NOT status EQUAL 1)
    list(APPEND failures "tlc.sh exited with '${status}', not 1")
endif()
if(NOT output MATCHES "Hang\\.cfg: FAIL, TLC did not finish within 1s \\(TLC_TIMEOUT\\)")
    list(APPEND failures "tlc.sh did not report that TLC ran out of time")
endif()
if(elapsed GREATER 10)
    list(APPEND failures "tlc.sh took ${elapsed}s to stop a run limited to 1s")
endif()
if(failures)
    list(JOIN failures "\n  " failures)
    message(FATAL_ERROR
        "Without timeout on PATH, tlc.sh must stop TLC at TLC_TIMEOUT even when the timer fires before the "
        "forked child has made its own process group:\n  ${failures}\ntlc.sh printed:\n${output}")
endif()
message(STATUS "tlc.sh stopped a hung TLC after ${elapsed}s with TLC_TIMEOUT=1 and no timeout on PATH.")
