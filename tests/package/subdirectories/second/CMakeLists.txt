if(NOT TARGET Rachis::RHI)
    find_package(Rachis 0.4 CONFIG REQUIRED)
endif()
add_fill_library(RachisShaderSecond)
